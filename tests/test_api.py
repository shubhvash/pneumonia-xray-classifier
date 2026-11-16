"""API tests (in-process, no server needed): uv run pytest -q"""
import io

from fastapi.testclient import TestClient
from PIL import Image

from predict import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_pneumonia_image():
    with open("data/chest_xray/INFECTED/person100_bacteria_475.jpeg", "rb") as f:
        r = client.post("/predict", files={"file": ("x.jpeg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["pneumonia_probability"] <= 1.0
    assert body["prediction"] in {"NORMAL", "PNEUMONIA"}


def test_predict_accepts_png_and_rgb():
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), "gray").save(buf, format="PNG")
    r = client.post("/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200


def test_rejects_non_image():
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
