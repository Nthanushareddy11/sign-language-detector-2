"""
Sign Language Detector
-----------------------
YOLOv8 trained on 26 ASL alphabet signs: A-Z
"""

from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "runs/train/sign_detector/weights/best.pt"

ASL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Unique colour per letter
COLOURS = {}
for i, c in enumerate(ASL_CLASSES):
    h = i / 26
    r = int(abs(h * 6 - 3) - 1) * 255
    g = int(2 - abs(h * 6 - 2)) * 255
    b = int(2 - abs(h * 6 - 4)) * 255
    COLOURS[c] = (max(0,min(255,b)), max(0,min(255,g)), max(0,min(255,r)))

def load_model() -> YOLO:
    if MODEL_PATH.exists():
        print(f"✅ Loading 26-class A-Z model from {MODEL_PATH}")
        return YOLO(str(MODEL_PATH))
    print("⚠️  No trained model found.")
    return YOLO("yolov8n.pt")


def detect_signs(model: YOLO, image_bytes: bytes, conf: float = 0.15) -> dict:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"detections": [], "text": "", "annotated": ""}

    results = model.predict(source=img, conf=conf, verbose=False)[0]

    detections = []
    letters    = []

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id] if cls_id < len(model.names) else str(cls_id)
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        detections.append({
            "sign"      : cls_name,
            "emoji"     : cls_name,
            "confidence": round(conf_val, 3),
            "bbox"      : [x1, y1, x2, y2],
        })
        letters.append(cls_name)

        colour = COLOURS.get(cls_name, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
        label  = f"{cls_name} {conf_val:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x1, y1 - lh - 14), (x1 + lw + 8, y1), colour, -1)
        cv2.putText(img, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    text = "".join(letters)
    if text:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h-55), (w, h), (0,0,0), -1)
        cv2.putText(img, f"Sign: {text}", (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,212,170), 3)

    _, buf = cv2.imencode(".jpg", img)
    return {
        "detections": detections,
        "text"      : text,
        "annotated" : base64.b64encode(buf.tobytes()).decode(),
    }
