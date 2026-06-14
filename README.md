# 🤟 Sign Language Detection System

Real-time **ASL Sign Language** detection using **YOLOv8 + FastAPI**.

Detects 5 common signs: **hello | iloveyou | no | thankyou | yes**

Dataset is **bundled** — no external download needed!

---

## 🚀 Quick Start

### 1. Install
```bash
pip3 install -r requirements.txt
```

### 2. Train (dataset already included!)
```bash
python3 model/train.py
```
- Pretrained weights (~6MB) download automatically on first run
- Takes ~15 min on Mac M1
- Best model saved to `runs/train/sign_detector/weights/best.pt`

### 3. Run API
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Open Frontend
Double-click `index.html` → click **Start Camera** → show hand signs! 🤟

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model info |
| POST | `/detect` | Detect sign from image |
| POST | `/detect/webcam` | Detect from base64 frame |

### Example Response
```json
{
  "detections": [
    {
      "sign": "hello",
      "emoji": "👋",
      "confidence": 0.934,
      "bbox": [120, 45, 380, 310]
    }
  ],
  "text": "hello",
  "latency_ms": 87.4
}
```

---

## 🎯 Signs Supported

| Sign | Emoji | Description |
|------|-------|-------------|
| hello | 👋 | Wave gesture |
| iloveyou | 🤟 | ILY hand sign |
| no | 🚫 | Head/hand shake |
| thankyou | 🙏 | Flat hand from chin |
| yes | ✅ | Fist nod |

---

## 🛠 Tech Stack
- **Model:** YOLOv8n (Ultralytics + PyTorch)
- **API:** FastAPI + Uvicorn
- **Vision:** OpenCV
- **Dataset:** 756 labelled images (bundled)
- **Container:** Docker

---

## 🐳 Docker
```bash
docker build -t sign-detector .
docker run -p 8000:8000 sign-detector
```

---

## 📝 Resume Bullet
> *"Built real-time ASL sign language detection system using YOLOv8 and PyTorch; trained on 756 labelled images across 5 sign classes; served via FastAPI with live webcam inference UI"*
