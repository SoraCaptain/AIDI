# app/memory/image_embedding_service.py

import io
import os
from typing import List
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer


class ImageEmbeddingService:
    """
    本地图像 embedding 服务。

    默认使用 sentence-transformers/clip-ViT-B-32。
    支持：
    - 本地图片路径
    - HTTP/HTTPS 图片 URL

    后续可以替换成：
    - OpenCLIP
    - SigLIP
    - DINOv2
    - 自研视觉 embedding service
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def load_image(self, image_path_or_url: str) -> Image.Image:
        if self._is_url(image_path_or_url):
            resp = requests.get(image_path_or_url, timeout=20)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return image

        if not os.path.exists(image_path_or_url):
            raise FileNotFoundError(f"Image not found: {image_path_or_url}")

        return Image.open(image_path_or_url).convert("RGB")

    def embed_image(self, image_path_or_url: str) -> List:
        image = self.load_image(image_path_or_url)

        vector = self.model.encode(
            image,
            normalize_embeddings=True,
        )

        return vector.astype(float).tolist()

    def cosine_similarity(
        self,
        query_vector: List[float],
        doc_vector: List[float],
    ) -> float:
        q = np.array(query_vector, dtype=np.float32)
        d = np.array(doc_vector, dtype=np.float32)

        if np.linalg.norm(q) == 0 or np.linalg.norm(d) == 0:
            return 0.0

        return float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d)))

    def _is_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in ["http", "https"]
