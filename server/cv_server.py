# cv_server.py
# uvicorn cv_server:app --host 0.0.0.0 --port 8200
import os

# Must be set BEFORE any paddle/paddlex imports to avoid:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import io
import time
import json
from typing import Optional, List
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, Query
from pydantic import BaseModel


app = FastAPI(title="Vision CV Server", version="0.2")


# -----------------------------
# Config
# -----------------------------

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "")
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")
DEVICE = os.getenv("CV_DEVICE", "cuda")

GROUNDING_DINO_CONFIG = os.getenv("GROUNDING_DINO_CONFIG", "")
GROUNDING_DINO_CHECKPOINT = os.getenv("GROUNDING_DINO_CHECKPOINT", "")


# -----------------------------
# Lazy model holders
# -----------------------------

_yolo_model = None
_ocr_model = None
_sam_mask_generator = None
_grounding_model = None


# -----------------------------
# Request Schemas
# -----------------------------

class ImageRequest(BaseModel):
    image_path: str


class OCRRequest(BaseModel):
    image_path: str


class YOLORequest(BaseModel):
    image_path: str
    conf: float = 0.25
    imgsz: int = 640
    classes: Optional[List[int]] = None


class SAMAutoRequest(BaseModel):
    image_path: str
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    max_masks: int = 30


class GroundingRequest(BaseModel):
    image_path: str
    text_prompt: str
    box_threshold: float = 0.35
    text_threshold: float = 0.25


# -----------------------------
# Utilities
# -----------------------------

def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ["http", "https"]


def load_image_pil(image_path: str) -> Image.Image:
    if is_url(image_path):
        resp = requests.get(image_path, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def load_image_cv2_bgr(image_path: str) -> np.ndarray:
    pil = load_image_pil(image_path)
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def load_image_rgb_np(image_path: str) -> np.ndarray:
    pil = load_image_pil(image_path)
    return np.array(pil)


def now_ms(start: float) -> float:
    return round((time.time() - start) * 1000, 2)


# -----------------------------
# Basic tools
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo_model": YOLO_MODEL_PATH,
        "device": DEVICE,
    }


@app.post("/inspect")
def inspect_image(req: ImageRequest):
    start = time.time()
    img = load_image_pil(req.image_path)

    return {
        "image_path": req.image_path,
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "latency_ms": now_ms(start),
    }


@app.post("/blur")
def detect_blur(req: ImageRequest):
    start = time.time()

    img = load_image_cv2_bgr(req.image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "image_path": req.image_path,
        "blur_score": float(score),
        "is_blurry": bool(score < 100),
        "threshold": 100,
        "latency_ms": now_ms(start),
    }


def get_ocr_model():
    global _ocr_model

    if _ocr_model is None:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x 的参数可能随版本变化；
        # 如果你的版本参数不同，先用 PaddleOCR() 最小初始化。
        try:
            _ocr_model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                # Disable MKLDNN/oneDNN to avoid PIR conversion bug in PaddlePaddle 3.3.x:
                #   NotImplementedError: ConvertPirAttribute2RuntimeAttribute
                #   not support [pir::ArrayAttribute<pir::DoubleAttribute>]
                enable_mkldnn=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize PaddleOCR: {e}\n"
                "Ensure paddleocr and paddlex are installed: pip install paddleocr paddlex\n"
                "On first run, model files are downloaded from HuggingFace — check network connectivity."
            ) from e

    return _ocr_model


@app.post("/ocr")
def ocr_image(req: OCRRequest):
    """
    OCR image text detection + recognition.

    注意：
    PaddleOCR 不同版本的 result 对象格式可能不同，
    这里做了兼容式解析。
    """
    start = time.time()
    ocr = get_ocr_model()

    result = ocr.predict(req.image_path)

    parsed = []

    for idx, res in enumerate(result):
        item = {
            "index": idx,
            "raw_type": str(type(res)),
        }

        if hasattr(res, "json"):
            try:
                item["json"] = res.json
            except Exception:
                pass

        if hasattr(res, "to_dict"):
            try:
                item["dict"] = res.to_dict()
            except Exception:
                pass

        if not item.get("json") and not item.get("dict"):
            item["repr"] = repr(res)

        parsed.append(item)

    return {
        "image_path": req.image_path,
        "ocr_results": parsed,
        "latency_ms": now_ms(start),
    }


def get_yolo_model():
    global _yolo_model

    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_PATH)

    return _yolo_model


@app.post("/yolo/detect")
def detect_objects_yolo(req: YOLORequest):
    start = time.time()

    model = get_yolo_model()

    results = model(
        req.image_path,
        conf=req.conf,
        imgsz=req.imgsz,
        classes=req.classes,
        verbose=False,
    )

    output = []

    for result in results:
        names = result.names
        boxes = result.boxes

        if boxes is None:
            continue

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
            conf = float(boxes.conf[i].detach().cpu().item())
            cls_id = int(boxes.cls[i].detach().cpu().item())
            cls_name = names.get(cls_id, str(cls_id))

            output.append(
                {
                    "bbox_xyxy": [round(float(x), 2) for x in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
            )

    return {
        "image_path": req.image_path,
        "model": YOLO_MODEL_PATH,
        "detections": output,
        "count": len(output),
        "latency_ms": now_ms(start),
    }


def get_sam_mask_generator(
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
):
    """
    为简单起见，这里每次参数变化不重建 generator。
    生产中建议根据参数缓存多个 generator，或固定参数。
    """
    global _sam_mask_generator

    if _sam_mask_generator is None:
        if not SAM_CHECKPOINT:
            raise ValueError(
                "SAM_CHECKPOINT is not set. "
                "Please set env SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth"
            )

        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device=DEVICE)

        _sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

    return _sam_mask_generator


@app.post("/sam/segment_auto")
def segment_with_sam_auto(req: SAMAutoRequest):
    start = time.time()

    image = load_image_rgb_np(req.image_path)

    mask_generator = get_sam_mask_generator(
        points_per_side=req.points_per_side,
        pred_iou_thresh=req.pred_iou_thresh,
        stability_score_thresh=req.stability_score_thresh,
    )

    all_masks = mask_generator.generate(image)

    masks = sorted(all_masks, key=lambda x: x.get("area", 0), reverse=True)
    masks = masks[: req.max_masks]

    summary = []

    for i, m in enumerate(masks):
        summary.append(
            {
                "index": i,
                "area": int(m.get("area", 0)),
                "bbox_xywh": [round(float(x), 2) for x in m.get("bbox", [])],
                "predicted_iou": float(m.get("predicted_iou", 0.0)),
                "stability_score": float(m.get("stability_score", 0.0)),
            }
        )

    return {
        "image_path": req.image_path,
        "mask_count_total": len(all_masks),
        "returned_count": len(summary),
        "masks": summary,
        "latency_ms": now_ms(start),
    }


# def get_grounding_model():
#     global _grounding_model

#     if _grounding_model is None:
#         if not GROUNDING_DINO_CONFIG or not GROUNDING_DINO_CHECKPOINT:
#             raise ValueError(
#                 "GROUNDING_DINO_CONFIG or GROUNDING_DINO_CHECKPOINT is not set."
#             )

#         from groundingdino.util.inference import load_model

#         _grounding_model = load_model(
#             GROUNDING_DINO_CONFIG,
#             GROUNDING_DINO_CHECKPOINT,
#         )

#     return _grounding_model


# @app.post("/grounding/detect")
# def grounding_detect(req: GroundingRequest):
#     start = time.time()

#     try:
#         from groundingdino.util.inference import load_image, predict
#     except Exception as e:
#         return {
#             "error": "GroundingDINO is not installed or import failed.",
#             "detail": repr(e),
#         }

#     model = get_grounding_model()

#     # GroundingDINO 的 load_image 通常接本地路径；
#     # 如果是 URL，先下载到临时文件。
#     image_path = req.image_path

#     if is_url(image_path):
#         os.makedirs("/tmp/vision_agent", exist_ok=True)
#         tmp_path = "/tmp/vision_agent/grounding_input.jpg"
#         resp = requests.get(image_path, timeout=30)
#         resp.raise_for_status()
#         with open(tmp_path, "wb") as f:
#             f.write(resp.content)
#         image_path = tmp_path

#     image_source, image = load_image(image_path)

#     boxes, logits, phrases = predict(
#         model=model,
#         image=image,
#         caption=req.text_prompt,
#         box_threshold=req.box_threshold,
#         text_threshold=req.text_threshold,
#     )

#     detections = []

#     for i in range(len(boxes)):
#         box = boxes[i].detach().cpu().numpy().tolist()
#         logit = float(logits[i].detach().cpu().item())
#         phrase = phrases[i]

#         detections.append(
#             {
#                 "box_cxcywh_norm": [round(float(x), 4) for x in box],
#                 "confidence": round(logit, 4),
#                 "phrase": phrase,
#             }
#         )

#     return {
#         "image_path": req.image_path,
#         "text_prompt": req.text_prompt,
#         "detections": detections,
#         "count": len(detections),
#         "latency_ms": now_ms(start),
#     }
