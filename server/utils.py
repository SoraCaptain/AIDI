import io
import os
import cv2
import base64
import requests
import numpy as np
from PIL import Image
from urllib.parse import urlparse


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ["http", "https"]


def is_data_url(value: str) -> bool:
    return value.startswith("data:")


def img_path_preprocess(image_path: str) -> str:
    if is_data_url(image_path):
        # data:image/xxx;base64,<b64>
        _, encoded = image_path.split(",", 1)
        img_data = base64.b64decode(encoded)

        os.makedirs("/tmp/vision_agent", exist_ok=True)
        tmp_path = "/tmp/vision_agent/grounding_input.jpg"
        with open(tmp_path, "wb") as f:
            f.write(img_data)
        image_path = tmp_path

    elif is_url(image_path):
        os.makedirs("/tmp/vision_agent", exist_ok=True)
        tmp_path = "/tmp/vision_agent/grounding_input.jpg"

        resp = requests.get(image_path, timeout=30)
        resp.raise_for_status()

        with open(tmp_path, "wb") as f:
            f.write(resp.content)

        image_path = tmp_path

    return image_path


def load_image_pil(image_path: str) -> Image.Image:
    if is_data_url(image_path):
        # data:image/xxx;base64,<b64>
        _, encoded = image_path.split(",", 1)
        img_data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(img_data)).convert("RGB")

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
