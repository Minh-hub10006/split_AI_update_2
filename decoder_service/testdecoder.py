import torch
import pika
import pickle
import numpy as np
import cv2
from src.core.yaml_config import YAMLConfig

# ==============================
# CONFIG
# ==============================
COCO_CLASSES = [
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
"traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
"dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
"baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
"broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
"bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
"microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
"teddy bear","hair drier","toothbrush"
]

# chỉ giữ người + phương tiện
TARGET_CLASSES = {0,1,2,3,5,7}
THRESHOLD = 0.5
QUEUE_NAME = "feature_queue"

# ==============================
# DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Decoder device:", device)

# ==============================
# LOAD MODEL
# ==============================

cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model
decoder = model.decoder

decoder.load_state_dict(
    torch.load("weight/decoder.pth", map_location=device)
)

decoder.to(device)
decoder.eval()

# ==============================
# RABBITMQ
# ==============================

connection = pika.BlockingConnection(
    pika.ConnectionParameters("localhost")
)

channel = connection.channel()

# ✅ Thêm durable=True để khớp với producer
channel.queue_declare(queue=QUEUE_NAME, durable=False)

# ==============================
# DECODER
# ==============================

def run_decoder(feats):
    with torch.no_grad():
        return decoder(feats)

# ==============================
# DRAW BOXES
# ==============================

def draw_boxes(frame, boxes, scores, labels=None, threshold=THRESHOLD):

    h, w = frame.shape[:2]

    count = 0

    for i, (box, score) in enumerate(zip(boxes, scores)):

        if score < threshold:
            continue

        cx, cy, bw, bh = box

        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))

        # Bỏ qua box quá nhỏ
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cls = int(labels[i])

        # bỏ các class không phải người hoặc xe
        if cls not in TARGET_CLASSES:
            continue

        class_name = COCO_CLASSES[cls]
        label_text = f"{class_name} {score:.2f}"
        

        cv2.putText(
            frame, label_text,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )

        count += 1

    return frame, count

# ==============================
# CALLBACK
# ==============================

def callback(ch, method, properties, body):
    try:
        payload = pickle.loads(body)

        frame_id  = payload["frame_id"]
        feats_np  = payload["features"]
        frame_jpg = payload["frame_jpg"]

        # ✅ Convert numpy → tensor, phục hồi batch dim
        feats = []
        for f in feats_np:

            t = torch.from_numpy(f).float().to(device)

            # Nếu producer đã squeeze batch dim thì unsqueeze lại
            if t.dim() == 3:
                t = t.unsqueeze(0)  # (C,H,W) → (1,C,H,W)

            feats.append(t)

        # ✅ Chạy decoder
        outputs = run_decoder(feats)

        # ✅ Lấy scores + boxes
        # pred_logits: (1, num_queries, num_classes)
        # pred_boxes:  (1, num_queries, 4)
        logits = outputs["pred_logits"][0]        # (num_queries, num_classes)
        boxes  = outputs["pred_boxes"][0].cpu().numpy()

        probs  = logits.softmax(-1)               # softmax qua classes
        scores = probs[:, :-1].max(-1)[0]         # bỏ class "no object" (class cuối)
        labels = probs[:, :-1].max(-1)[1]         # class index

        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        # ✅ Vẽ lên frame
        frame = cv2.imdecode(
            np.frombuffer(frame_jpg, np.uint8),
            cv2.IMREAD_COLOR
        )

        frame, count = draw_boxes(frame, boxes, scores, labels)

        print(f"✅ Frame {frame_id} | {count} detections")

        cv2.imshow("Detection", frame)

        # ✅ Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("🛑 Người dùng thoát")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            raise KeyboardInterrupt

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except KeyboardInterrupt:
        raise

    except Exception as e:
        import traceback
        print("❌ Lỗi:", str(e))
        traceback.print_exc()
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

# ==============================
# START
# ==============================

print("🎯 Decoder ready, waiting for frames...")

channel.basic_qos(prefetch_count=10)

channel.basic_consume(
    queue=QUEUE_NAME,
    on_message_callback=callback
)

try:
    channel.start_consuming()

except KeyboardInterrupt:
    print("\n⏹️ Dừng decoder")

finally:
    cv2.destroyAllWindows()
    connection.close()
    print("✅ Đã đóng kết nối")