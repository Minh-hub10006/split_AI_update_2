import torch
import pika
import cv2
import pickle
import torchvision.transforms as T
from PIL import Image
import numpy as np
from src.core.yaml_config import YAMLConfig
import os
import time
import torch.nn.functional as F
from queue import Queue
from threading import Thread

# ==============================
# CONFIG
# ==============================

VIDEO_PATH = "traffic.mp4"
QUEUE_NAME = "feature_queue"

SMALL_FRAME = (1280,720)
FEATURE_SIZE = (8,8)

JPG_QUALITY = 60

FRAME_QUEUE_SIZE = 30
PAYLOAD_QUEUE_SIZE = 30

print(f"🎥 Streaming từ {VIDEO_PATH}")

# ==============================
# VIDEO
# ==============================

if not os.path.exists(VIDEO_PATH):
    print("❌ Không tìm thấy video")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Không mở được video")
    exit()

# ==============================
# PREPROCESS
# ==============================

transform = T.Compose([
    T.Resize((640,640)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def preprocess(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(frame_rgb)

    tensor = transform(pil).unsqueeze(0)

    return tensor


# ==============================
# JPEG COMPRESS
# ==============================

def compress_frame(frame):

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]

    ok, buffer = cv2.imencode(".jpg", frame, encode_param)

    if not ok:
        raise RuntimeError("JPG encode failed")

    return buffer.tobytes()


# ==============================
# RABBITMQ
# ==============================

connection = pika.BlockingConnection(
    pika.ConnectionParameters("localhost")
)

channel = connection.channel()

channel.queue_declare(queue=QUEUE_NAME, durable=False)

# ==============================
# LOAD MODEL
# ==============================

cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")

model = cfg.model

backbone = model.backbone
encoder = model.encoder

backbone.load_state_dict(
    torch.load("weight/backbone.pth", map_location="cpu")
)

encoder.load_state_dict(
    torch.load("weight/encoder.pth", map_location="cpu")
)

backbone.eval()
encoder.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone.to(device)
encoder.to(device)
# ==============================
# ENCODER
# ==============================

# ==============================
# ENCODER
# ==============================

def run_encoder(frame_tensor):

    with torch.no_grad():

        # B1: Backbone → feature pyramid (P2, P3, P4, P5)
        feats = backbone(frame_tensor)

        # B2: Encoder xử lý feature pyramid
        encoder_out = encoder(feats)

        # B3: Convert sang numpy để pickle + gửi qua RabbitMQ
        processed_feats = []

        for f in encoder_out:

            f = f.float()

            # Bỏ batch dim nếu batch_size = 1
            if f.dim() == 4 and f.shape[0] == 1:
                f = f.squeeze(0)   # (C, H, W)

            processed_feats.append(f.cpu().numpy())

        return processed_feats


# ==============================
# QUEUES
# ==============================

frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
payload_queue = Queue(maxsize=PAYLOAD_QUEUE_SIZE)

# ==============================
# STATS
# ==============================

frame_id = 0
total_frames = 0
total_kb = 0

start_time = time.time()

# ==============================
# VIDEO READER THREAD
# ==============================

def video_reader():

    global frame_id

    while True:

        ret, frame = cap.read()

        if not ret:
            print("🔁 Loop video")
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            continue

        frame_queue.put((frame_id,frame))

        frame_id += 1


# ==============================
# ENCODER THREAD
# ==============================

def encoder_worker():
    global frame_id  # Add this if needed
    while True:
        fid, frame = frame_queue.get()
        
        tensor = preprocess(frame).to(device)
        
        # 🔥 USE THE FIXED FUNCTION
        features = run_encoder(tensor)  
        
        print(f"📤 Frame {fid}: {len(features)} feature levels")
        
        small = cv2.resize(frame, SMALL_FRAME)
        frame_jpg = compress_frame(small)

        payload = {
            "frame_id": fid,
            "timestamp": time.time(),
            "features": features,  
            "frame_jpg": frame_jpg,
            "orig_shape": frame.shape,
            "small_shape": small.shape
        }
        payload_queue.put(payload)


# ==============================
# SENDER THREAD
# ==============================

def sender_worker():

    global total_frames
    global total_kb

    while True:

        payload = payload_queue.get()

        data = pickle.dumps(payload)

        channel.basic_publish(
            exchange="",
            routing_key="feature_queue",
            body=data,
            properties=pika.BasicProperties(delivery_mode=2)
        )

        total_frames += 1
        

        print(f"[{payload['frame_id']:4d}] KB sent")


# ==============================
# START THREADS
# ==============================

print("🚀 Start threaded streaming")

reader = Thread(target=video_reader, daemon=True)
encoder_t = Thread(target=encoder_worker, daemon=True)
sender = Thread(target=sender_worker, daemon=True)

reader.start()
encoder_t.start()
sender.start()

# ==============================
# MAIN LOOP
# ==============================

try:

    while True:
        time.sleep(1)

except KeyboardInterrupt:

    print("\n⏹️ Stop streaming")

finally:
    cap.release()

    # Đợi các thread xử lý nốt
    time.sleep(2)

    try:
        connection.close()
    except Exception:
        pass  # Bỏ qua lỗi đóng connection

    

    print("\n" + "="*60)
    print("🏁 STREAM FINISHED")
    