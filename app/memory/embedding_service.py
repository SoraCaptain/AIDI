# app/memory/embedding_service.py

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    本地 embedding 服务。

    默认使用中文友好的 bge-small-zh-v1.5。
    后续可以替换成：
    - bge-m3
    - 本地 vLLM embedding endpoint
    - TEI
    - 自研 embedding service
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def embed_text(self, text: str) -> List[float]:
        if not text:
            text = ""

        vector = self.model.encode(
            text,
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
