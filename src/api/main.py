import io
import os
from contextlib import asynccontextmanager

import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.model import CLASS_NAMES, IMAGE_SIZE

MODEL_PATH = os.getenv("MODEL_PATH", "models/human_face_v1.keras")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB limit
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    # Load natively with Keras 3
    app.state.model = keras.models.load_model(MODEL_PATH)
    yield
    app.state.model = None


app = FastAPI(
    title="Humanly - Face Classifier API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# Allow any frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def serve_ui():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Humanly API is online", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "classes": CLASS_NAMES,
        "image_size": list(IMAGE_SIZE),
        "threshold": THRESHOLD,
    }


def run_prediction(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    pixels = np.asarray(image, dtype=np.float32)[None, ...]

    p_non_human = float(model.predict(pixels, verbose=0)[0][0])
    predicted_index = int(p_non_human >= THRESHOLD)
    return {
        "prediction": CLASS_NAMES[predicted_index],
        "human_probability": round(1.0 - p_non_human, 4),
        "non_human_probability": round(p_non_human, 4),
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller")

    try:
        result = await run_in_threadpool(run_prediction, app.state.model, image_bytes)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image")
    finally:
        await file.close()

    return result