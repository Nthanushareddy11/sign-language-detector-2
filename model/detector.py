"""
Sign Language Detector
-----------------------
Supports TWO models:
  - Word model  : hello, iloveyou, no, thankyou, yes
  - Alpha model : A-Z (26 letters)
"""

from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path

ROOT = Path(__file__).parent.parent

WORD_MODEL_PATH  = ROOT / "runs/train/sign_detector/weights/word_best.pt"
ALPHA_MODEL_PATH = ROOT / "runs/train/sign_detector/weights/alpha_best.pt"

WORD_EMOJIS = {
    "hello"    : "👋",
    "iloveyou" : "🤟",
    "no"       : "🚫",
    "thankyou" : "🙏",
    "yes"      : "✅",
}

WORD_COLOURS = {
    "hello"    : (0, 212, 170),
    "iloveyou" : (255, 111, 145),
    "no"       : (248, 113, 113),
    "thankyou" : (74, 222, 128),
    "yes"      : (251, 191, 36),
}

ALPHA_COLOURS = {
    'A':(255,99,71),'B':(255,165,0),'C':(255,215,0),'D':(154,205,50),
    'E':(0,205,0),'F':(0,212,170),'G':(0,191,255),'H':(30,144,255),
    'I':(138,43,226),'J':(255,20,147),'K':(255,99,71),'L':(255,165,0),
    'M':(255,215,0),'N':(154,205,50),'O':(0,205,0),'P':(0,212,170),
    'Q':(0,191,255),'R':(30,144,255),'S':(138,43,226),'T':(255,20,147),
    'U':(255,99,71),'V':(255,165,0),'W':(255,215,0),'X':(154,205,50),
    'Y':(0,205,0),'Z':(0,212,170),
}


def load_models() -> dict:
    models = {}
    if WORD_MODEL_PATH.exists():
        print(f"✅ Loading word model from {WORD_MODEL_PATH}")
        models["word"] = YOLO(str(WORD_MODEL_PATH))
    elif (ROOT / "runs/train/sign_detector/weights/best.pt").exists():
        # fallback
        models["word"] = YOLO(str(ROOT / "runs/train/sign_detector/weights/best.pt"))
        print("✅ Word model loaded (fallback)")

    if ALPHA_MODEL_PATH.exists():
        print(f"✅ Loading alpha model from {ALPHA_MODEL_PATH}")
        models["alpha"] = YOLO(str(ALPHA_MODEL_PATH))
    else:
        print("⚠️  Alpha model not found")

    return models


def load_model():
    """Single model loader for backward compatibility."""
    models = load_models()
    return models.get("word") or models.get("alpha")


def detect_signs(model, image_bytes: bytes, conf: float = 0.15, mode: str = "word") -> dict:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"detections": [], "text": "", "annotated": ""}

    results = model.predict(source=img, conf=conf, verbose=False)[0]

    detections = []
    signs      = []

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id] if cls_id < len(model.names) else str(cls_id)
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        emoji = WORD_EMOJIS.get(cls_name, cls_name)
        colour = WORD_COLOURS.get(cls_name, ALPHA_COLOURS.get(cls_name, (0,255,0)))

        detections.append({
            "sign"      : cls_name,
            "emoji"     : emoji,
            "confidence": round(conf_val, 3),
            "bbox"      : [x1, y1, x2, y2],
        })
        signs.append(cls_name)

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
        label = f"{emoji} {cls_name} {conf_val:.0%}" if mode == "word" else f"{cls_name} {conf_val:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.rectangle(img, (x1, y1 - lh - 12), (x1 + lw + 8, y1), colour, -1)
        cv2.putText(img, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 2)

    text = " | ".join(signs) if mode == "word" else "".join(signs)
    if text:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h-55), (w, h), (0,0,0), -1)
        cv2.putText(img, text, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,212,170), 2)

    _, buf = cv2.imencode(".jpg", img)
    return {
        "detections": detections,
        "text"      : text,
        "annotated" : base64.b64encode(buf.tobytes()).decode(),
    }