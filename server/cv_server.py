# cv_server.py
# uvicorn cv_server:app --host 0.0.0.0 --port 8200
from fastapi import FastAPI
from PIL import Image
import cv2
import os
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect")
def inspect_image(path: str):
    img = Image.open(path)
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode
    }


@app.post("/blur")
def detect_blur(path: str):
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
    img = cv2.imread(path, 0)
    if img is None:
        return {"error": f"Cannot read image (corrupted or unsupported format): {path}"}
    score = cv2.Laplacian(img, cv2.CV_64F).var()
    return {
        "blur_score": float(score),
        "is_blurry": bool(score < 100)
    }