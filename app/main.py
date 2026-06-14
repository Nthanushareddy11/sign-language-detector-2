"""
Sign Language Detection API
----------------------------
Endpoints:
    GET  /           → health check
    GET  /health     → model info
    POST /detect     → detect sign language from image
    POST /detect/webcam → detect from base64 webcam frame
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import sys
import os
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.detector import load_model, detect_signs

# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Sign Language Detection API",
    description = "Real-time ASL (American Sign Language) hand sign detection using YOLOv8 + MediaPipe.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

model = None

@app.on_event("startup")
async def startup():
    global model
    print("Loading Sign Language Detection model...")
    model = load_model()
    print("✅ Model loaded!")


class WebcamInput(BaseModel):
    image: str   # base64 encoded image


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Sign Language Detection API is running 🤟", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status"  : "healthy",
        "model"   : "YOLOv8 + MediaPipe",
        "classes" : list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"],
        "total"   : 29
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image and detect ASL hand signs.

    Returns
    -------
    - detections  : list of detected signs with confidence + bbox
    - text        : detected letters joined as text
    - annotated   : base64 annotated image
    - latency_ms  : inference time
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    start = time.time()
    result = detect_signs(model, image_bytes)
    latency = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "filename"   : file.filename,
        "detections" : result["detections"],
        "text"       : result["text"],
        "annotated"  : result["annotated"],
        "latency_ms" : latency,
    })


@app.post("/detect/webcam")
async def detect_webcam(input: WebcamInput):
    """Detect sign language from a base64 encoded webcam frame."""
    try:
        image_bytes = base64.b64decode(input.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image.")

    start = time.time()
    result = detect_signs(model, image_bytes)
    latency = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "detections" : result["detections"],
        "text"       : result["text"],
        "annotated"  : result["annotated"],
        "latency_ms" : latency,
    })
