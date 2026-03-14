import pika, cv2, pickle
import os

video_path = "traffic.mp4"
if not os.path.exists(video_path):
    print(f"❌ File {video_path} không tồn tại!")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Không mở được video!")
    exit()

print(f"📹 Đang đọc {video_path}")

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="frames_queue", durable=True)

# ✅ THÊM: Queue raw frames
channel.queue_declare(queue="frame_queue_raw", durable=True)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Hết video")
        break
    
    if frame is None or frame.size == 0:
        print(f"⚠️ Frame {frame_id} rỗng")
        continue
    
    print(f"Frame {frame_id}: {frame.shape}")
    
    # ✅ SỬA: Gửi 2 queues
    payload_enc = {"frame_id": frame_id, "frame": frame}
    data_enc = pickle.dumps(payload_enc)
    channel.basic_publish(exchange="", routing_key="frames_queue", body=data_enc)
    
    # ✅ THÊM: Raw queue
    payload_raw = {"frame_id": frame_id, "frame": frame}
    data_raw = pickle.dumps(payload_raw)
    channel.basic_publish(exchange="",routing_key="frame_queue_raw", body=data_raw)
    
    print(f"📤 Frame {frame_id} → encoder+raw, {len(data_enc)/1024:.1f}KB")
    frame_id += 1

cap.release()
connection.close()
print(f"🎉 Đã gửi {frame_id} frames!")