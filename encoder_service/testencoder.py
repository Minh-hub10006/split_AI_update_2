import torch
import pika
import cv2
import pickle
import torchvision.transforms as T
from PIL import Image
import numpy as np
from src.core.yaml_config import YAMLConfig

# Load model
cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model
backbone = model.backbone
encoder = model.encoder

backbone.load_state_dict(torch.load("weight/backbone.pth", map_location="cpu"))
encoder.load_state_dict(torch.load("weight/encoder.pth", map_location="cpu"))
backbone.eval()
encoder.eval()

transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def frame_to_tensor(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return transform(pil_img).unsqueeze(0)

def run_encoder(frame_tensor):
    with torch.no_grad():
        feats = backbone(frame_tensor)
        memory = encoder(feats)
    return memory

def send_feature_payload(payload):
    conn = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
    ch = conn.channel()
    ch.queue_declare(queue="feature_queue", durable=False, passive=True)
    data = pickle.dumps(payload)
    ch.basic_publish(exchange="", routing_key="feature_queue", body=data)
    conn.close()

# Connection chính để nhận frames
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="frames_queue", durable=True)

def callback(ch, method, properties, body):
    try:
        payload = pickle.loads(body)
        frame_id = payload["frame_id"]
        frame = payload["frame"]
        
        # Check frame hợp lệ
        if frame is None or frame.size == 0:
            print(f"⚠️ Skip empty frame {frame_id}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        print(f"📥 Frame {frame_id}: {frame.shape}")
        
        # Process
        x = frame_to_tensor(frame)
        memory = run_encoder(x)
        
        # ✅ FIX: Xử lý list tensor
        if isinstance(memory, list):
            memory_np = [m.detach().cpu().numpy() for m in memory]
        else:
            memory_np = memory.detach().cpu().numpy()
        
        out_payload = {
            "frame_id": frame_id,
            "feature": memory_np,
            "shape": frame.shape,
        }
        
        send_feature_payload(out_payload)
        print(f"✅ Sent feature {frame_id}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        print(f"❌ Error frame {frame_id}: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

print("🎯 Encoder ready - listening frames_queue...")
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue="frames_queue", on_message_callback=callback)
channel.start_consuming()