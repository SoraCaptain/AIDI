
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel
from utils import img_path_preprocess

from groundingdino.util.inference import load_model, load_image, predict


app = FastAPI(title="GroundingDINO Server")

CONFIG_PATH = os.getenv(
    "GROUNDING_DINO_CONFIG",
    "/mnt/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
)

CHECKPOINT_PATH = os.getenv(
    "GROUNDING_DINO_CHECKPOINT",
    "/mnt/models/GroundingDINO/weights/groundingdino_swint_ogc.pth",
)

_model = None


class GroundingRequest(BaseModel):
    image_path: str
    text_prompt: str
    box_threshold: float = 0.35
    text_threshold: float = 0.25


def get_model():
    global _model

    if _model is None:
        _model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

    return _model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "config": CONFIG_PATH,
        "checkpoint": CHECKPOINT_PATH,
    }


@app.post("/grounding/detect")
def grounding_detect(req: GroundingRequest):
    start = time.time()

    image_path = img_path_preprocess(req.image_path)

    model = get_model()

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=req.text_prompt,
        box_threshold=req.box_threshold,
        text_threshold=req.text_threshold,
    )

    detections = []

    for i in range(len(boxes)):
        box = boxes[i].detach().cpu().numpy().tolist()
        logit = float(logits[i].detach().cpu().item())
        phrase = phrases[i]

        detections.append(
            {
                "box_cxcywh_norm": [round(float(x), 4) for x in box],
                "confidence": round(logit, 4),
                "phrase": phrase,
            }
        )

    return {
        "image_path": req.image_path,
        "text_prompt": req.text_prompt,
        "detections": detections,
        "count": len(detections),
        "latency_ms": round((time.time() - start) * 1000, 2),
    }
