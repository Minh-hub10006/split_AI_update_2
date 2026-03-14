import pika
import pickle
import cv2
import numpy as np
from collections import defaultdict
import atexit

raw_frames = {}  # frame_id → frame
predictions = {}  # frame_id → (boxes, scores)
video_writer = None

def draw_predictions(frame, boxes, scores, threshold=0.5):
    img = frame.copy()
    h, w = img.shape[:2]
    
    for box, score in zip(boxes, scores):
        if score < threshold: continue
        
        cx, cy, bw, bh = box
        x1 = max(0, int((cx-bw/2)*w))
        y1 = max(0, int((cy-bh/2)*h))
        x2 = min(w, int((cx+bw/2)*w))
        y2 = min(h, int((cy+bh/2)*h))
        
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(img, f'{score:.2f}', (x1,y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return img

def raw_frame_callback(ch, method, properties, body):
    payload = pickle.loads(body)
    frame_id = payload["frame_id"]
    frame = payload["frame"]
    
    raw_frames[frame_id] = frame
    print(f"🖼️ Raw frame {frame_id}")
    ch.basic_ack(delivery_tag=method.delivery_tag)
    
    # Check nếu có prediction → vẽ ngay
    if frame_id in predictions:
        process_frame(frame_id)

def pred_callback(ch, method, properties, body):
    payload = pickle.loads(body)
    frame_id = payload["frame_id"]
    boxes, scores = payload["boxes"], payload["scores"]
    
    predictions[frame_id] = (boxes, scores)
    print(f"🔍 Pred frame {frame_id}: {len(scores)} boxes")
    ch.basic_ack(delivery_tag=method.delivery_tag)
    
    # Check nếu có raw frame → vẽ ngay
    if frame_id in raw_frames:
        process_frame(frame_id)

def process_frame(frame_id):
    global video_writer
    
    # ✅ SAFE CHECK + POP (không del 2 lần)
    if frame_id not in raw_frames or frame_id not in predictions:
        return
    
    frame = raw_frames.pop(frame_id)      # POP thay DEL
    boxes, scores = predictions.pop(frame_id)
    
    print(f"🎨 Process frame {frame_id}")
    
    # Vẽ bbox
    result = draw_predictions(frame, boxes, scores)
    
    # Video writer
    if video_writer is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('final_output.avi', fourcc, 25.0, (w, h))
    
    video_writer.write(result)
    print(f"✅ Video frame {frame_id}")
def cleanup():
    global video_writer
    if video_writer:
        video_writer.release()
        print("🎥 VIDEO HOÀN THÀNH!")

atexit.register(cleanup)

# 2 CONNECTIONS
raw_conn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
pred_conn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))

raw_ch = raw_conn.channel()
pred_ch = pred_conn.channel()

raw_ch.queue_declare(queue='frame_queue_raw', passive=True)
pred_ch.queue_declare(queue='predict_queue', passive=True)

print("🎬 VideoBuilder ready...")
raw_ch.basic_consume('frame_queue_raw', raw_frame_callback)
pred_ch.basic_consume('predict_queue', pred_callback)

# Multi-thread consume
import threading
raw_ch.basic_qos(prefetch_count=1)
pred_ch.basic_qos(prefetch_count=1)

print("🎬 VideoBuilder ready...")
threading.Thread(target=raw_ch.start_consuming, daemon=True).start()
pred_ch.start_consuming()