"""
Train YOLOv8 on Sign Language Dataset.

Dataset is BUNDLED in data/sign_yolo/ — no download needed!
Classes: hello, iloveyou, no, thankyou, yes (5 classes)

On first run, YOLOv8n pretrained weights (~6MB) download automatically.
Transfer learning reaches ~95%+ mAP50 in ~15 min on Mac M1.

Usage
-----
    python model/train.py
"""

from ultralytics import YOLO
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_YAML = ROOT / "data" / "data.yaml"


def train():
    print("Loading YOLOv8n pretrained model...")
    model = YOLO("yolov8n.pt")   # downloads ~6MB on first run

    results = model.train(
        data     = str(DATA_YAML),
        epochs   = 60,
        imgsz    = 640,
        batch    = 16,
        project  = str(ROOT / "runs/train"),
        name     = "sign_detector",
        patience = 20,
        device   = "mps",    # Mac M1 — change to 0 for GPU, cpu for CPU
        seed     = 42,
    )

    best = ROOT / "runs/train/sign_detector/weights/best.pt"
    print(f"\n✅ Training complete! Best weights: {best}")
    m = results.results_dict
    print(f"   mAP50    : {m.get('metrics/mAP50(B)', 0):.3f}")
    print(f"   mAP50-95 : {m.get('metrics/mAP50-95(B)', 0):.3f}")
    return best


if __name__ == "__main__":
    train()
