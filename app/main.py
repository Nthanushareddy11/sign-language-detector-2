"""
Sign Language Detection API
----------------------------
Supports two modes:
  - word  : hello, iloveyou, no, thankyou, yes
  - alpha : A-Z (26 letters)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time, sys, os, base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.detector import load_models, detect_signs

app = FastAPI(
    title="Sign Language Detection API",
    description="Dual-mode ASL detection: word signs + A-Z alphabet using YOLOv8.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

models = {}

@app.on_event("startup")
async def startup():
    global models
    print("Loading Sign Language models...")
    models = load_models()
    print(f"✅ Loaded models: {list(models.keys())}")


class WebcamInput(BaseModel):
    image: str
    mode: str = "word"


@app.get("/")
def root():
    return {"message": "Sign Language Detection API 🤟", "docs": "/docs", "modes": ["word", "alpha"]}


@app.get("/health")
def health():
    return {
        "status"  : "healthy",
        "models"  : list(models.keys()),
        "word_signs" : ["hello", "iloveyou", "no", "thankyou", "yes"],
        "alpha_signs": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    mode: str = Query(default="word", enum=["word", "alpha"])
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    if mode not in models:
        available = list(models.keys())
        mode = available[0] if available else None
        if not mode:
            raise HTTPException(status_code=503, detail="No models loaded.")

    image_bytes = await file.read()
    start = time.time()
    result = detect_signs(models[mode], image_bytes, mode=mode)
    latency = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "filename"   : file.filename,
        "mode"       : mode,
        "detections" : result["detections"],
        "text"       : result["text"],
        "annotated"  : result["annotated"],
        "latency_ms" : latency,
    })


@app.post("/detect/webcam")
async def detect_webcam(input: WebcamInput):
    try:
        image_bytes = base64.b64decode(input.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image.")

    mode = input.mode if input.mode in models else list(models.keys())[0]

    start = time.time()
    result = detect_signs(models[mode], image_bytes, mode=mode)
    latency = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "mode"       : mode,
        "detections" : result["detections"],
        "text"       : result["text"],
        "annotated"  : result["annotated"],
        "latency_ms" : latency,
    })