"""FastAPI web service that serves the pneumonia classifier (ONNX Runtime).

Run locally:
    uv run uvicorn predict:app --host 0.0.0.0 --port 9696
Endpoints:
    GET  /          small upload page for manual testing in the browser
    GET  /health    liveness / readiness check
    POST /predict   multipart upload, field name "file" (JPEG/PNG chest X-ray)
    GET  /docs      interactive Swagger UI
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

meta = json.loads((MODEL_DIR / "metadata.json").read_text())
session = ort.InferenceSession(str(MODEL_DIR / meta["model_file"]), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
IMG_SIZE = meta["img_size"]
MEAN = np.array(meta["mean"], dtype=np.float32).reshape(3, 1, 1)
STD = np.array(meta["std"], dtype=np.float32).reshape(3, 1, 1)
THRESHOLD = float(meta["threshold"])

app = FastAPI(
    title="Pneumonia X-ray classifier",
    description="Classifies a pediatric chest X-ray as NORMAL or PNEUMONIA. "
    "Research/education demo — not a medical device.",
    version="1.0.0",
)


class Prediction(BaseModel):
    pneumonia_probability: float
    prediction: str
    threshold: float


def preprocess(image: Image.Image) -> np.ndarray:
    """Same steps as training (common.load_resized + ToTensor + Normalize)."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = (x - MEAN) / STD
    return x[np.newaxis, ...]


def predict_image(image: Image.Image) -> Prediction:
    logit = session.run(None, {INPUT_NAME: preprocess(image)})[0][0]
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    label = meta["class_names"][int(prob >= THRESHOLD)]
    return Prediction(pneumonia_probability=round(prob, 4), prediction=label, threshold=THRESHOLD)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": meta["model_file"], "test_metrics": meta.get("test_metrics")}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)")) -> Prediction:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return predict_image(image)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Pneumonia X-ray classifier</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
#out{font-size:1.2em;margin-top:16px}img{max-width:100%;margin-top:12px;border-radius:6px}</style></head>
<body><h2>Pneumonia X-ray classifier</h2>
<p>Upload a pediatric chest X-ray (JPEG/PNG). Demo only — not for clinical use.
API docs: <a href="/docs">/docs</a></p>
<input type="file" id="f" accept="image/*"><div id="out"></div><img id="preview">
<script>
document.getElementById('f').onchange = async (e) => {
  const file = e.target.files[0]; if (!file) return;
  document.getElementById('preview').src = URL.createObjectURL(file);
  const out = document.getElementById('out'); out.textContent = 'Predicting...';
  const fd = new FormData(); fd.append('file', file);
  const r = await fetch('/predict', {method: 'POST', body: fd});
  const j = await r.json();
  out.textContent = r.ok ? `${j.prediction} — P(pneumonia) = ${j.pneumonia_probability} (threshold ${j.threshold})`
                         : `Error: ${j.detail}`;
};
</script></body></html>"""
