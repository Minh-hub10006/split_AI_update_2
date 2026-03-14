import torch
import pika
import pickle
import cv2
import numpy as np
from src.core.yaml_config import YAMLConfig

cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model
decoder = model.decoder
decoder.load_state_dict(torch.load("weight/decoder.pth", map_location="cpu"))
decoder.eval()

def run_decoder(feats):
    with torch.no_grad():
        outputs = decoder(feats)
    return outputs

def send_to_predictor(payload):
    """Gửi predictions qua MQ"""
    conn = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
    ch = conn.channel()
    ch.queue_declare(queue="predict_queue")
    data = pickle.dumps(payload)
    ch.basic_publish(exchange="", routing_key="predict_queue", body=data)
    conn.close()

def callback(ch, method, properties, body):
    try:
        payload = pickle.loads(body)
        frame_id = payload["frame_id"]
        feats_np = payload["feature"]  # numpy từ encoder
        shape = payload["shape"]
        
        print(f"📥 Decoder frame {frame_id}")
        
        # Chuyển numpy → tensor
        if isinstance(feats_np, list):
            feats = [torch.from_numpy(f).float() for f in feats_np]
        else:
            feats = torch.from_numpy(feats_np).float()
        
        # Decode
        outputs = run_decoder(feats)
        pred_logits = outputs["pred_logits"][0]
        pred_boxes = outputs["pred_boxes"][0]
        
        scores = pred_logits.softmax(-1).max(-1)[0].cpu().numpy()
        boxes = pred_boxes.cpu().numpy()
        
        # Payload cho predictor
        pred_payload = {
            "frame_id": frame_id,
            "boxes": boxes,
            "scores": scores,
            "shape": shape,
        }
        
        send_to_predictor(pred_payload)
        print(f"📤 Sent predictions frame {frame_id}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        print(f"❌ Decoder error: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

# MQ setup
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="feature_queue")

print("🎯 Decoder ready - listening feature_queue...")
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue="feature_queue", on_message_callback=callback)
channel.start_consuming()