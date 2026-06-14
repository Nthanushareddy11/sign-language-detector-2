# 🤟 Sign Language Detection System

Real-time **ASL (American Sign Language)** detection using **YOLOv8 + FastAPI**.

Detects **26 letters A-Z** in real time via webcam.

🌐 **Live Demo:** [nthanushareddy11.github.io/sign-language-detector](https://nthanushareddy11.github.io/sign-language-detector)

---

## 🎯 Features

- 📷 **Live webcam detection** — show any hand sign A-Z
- 🔤 **Full alphabet** — detects all 26 ASL letters
- ✍️ **Word builder** — spell words letter by letter
- ⚡ **Real-time** — ~70ms inference latency
- 📁 **Image upload** — test with any photo

---

## 🖼️ Demo

| Webcam Detection | Word Builder |
|-----------------|--------------|
| Show hand sign → letter detected instantly | Spell words by signing letters one by one |

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Open Frontend
```bash
python3 -m http.server 3000
# Open http://127.0.0.1:3000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model info + 26 classes |
| POST | `/detect` | Detect sign from image |
| POST | `/detect/webcam` | Detect from base64 frame |

### Example Response
```json
{
  "detections": [
    {
      "sign": "A",
      "confidence": 0.91,
      "bbox": [120, 45, 380, 310]
    }
  ],
  "text": "A",
  "latency_ms": 68.4
}
```

---

## 🔤 Supported Signs

| A | B | C | D | E | F | G | H | I | J | K | L | M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

| N | O | P | Q | R | S | T | U | V | W | X | Y | Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 🛠 Tech Stack

- **Model:** YOLOv8n (Ultralytics)
- **Framework:** PyTorch
- **API:** FastAPI + Uvicorn
- **Vision:** OpenCV
- **Frontend:** Vanilla JS + HTML5 Canvas
- **Deployment:** Render (API) + GitHub Pages (Frontend)

---

## 🐳 Docker
```bash
docker build -t sign-detector .
docker run -p 8000:8000 sign-detector
```

---

## 📝 Resume Bullet
> *"Built real-time ASL sign language detection system using YOLOv8 and PyTorch; detects all 26 alphabet letters; served via FastAPI with live webcam inference and word-building UI; deployed on Render + GitHub Pages"*