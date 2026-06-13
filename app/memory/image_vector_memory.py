# app/memory/image_vector_memory.py

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.memory.image_embedding_service import ImageEmbeddingService


class ImageVectorMemory:
    """
    SQLite-based image vector memory.

    保存每张图片的 CLIP embedding。
    通过当前图片 embedding 检索历史相似图片。
    """

    def __init__(
        self,
        db_path: str = "data/memory/vision_memory.sqlite3",
        image_embedding_service: Optional[ImageEmbeddingService] = None,
    ):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.image_embedding_service = (
            image_embedding_service or ImageEmbeddingService()
        )

        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image_embeddings (
                    image_key TEXT PRIMARY KEY,
                    task_id TEXT,
                    image_path TEXT,
                    embedding_json TEXT,
                    created_at TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_image_embeddings_task_id
                ON image_embeddings(task_id)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_image_embeddings_created_at
                ON image_embeddings(created_at)
                """
            )

            conn.commit()

    def upsert_image_embedding(
        self,
        task_id: str,
        image_path: str,
    ):
        """
        为当前图片保存 embedding。

        image_key 暂时用 image_path。
        后续可以升级为：
        - image_id
        - file hash
        - perceptual hash
        """

        if not image_path:
            return

        embedding = self.image_embedding_service.embed_image(image_path)
        created_at = datetime.utcnow().isoformat()

        image_key = image_path

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO image_embeddings (
                    image_key,
                    task_id,
                    image_path,
                    embedding_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    image_key,
                    task_id,
                    image_path,
                    json.dumps(embedding),
                    created_at,
                ),
            )

            conn.commit()

    def search_similar_images(
        self,
        image_path: str,
        top_k: int = 5,
        min_score: float = 0.25,
        exclude_same_image: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        基于当前图片 embedding 检索历史相似图片。
        """

        if not image_path:
            return []

        query_vector = self.image_embedding_service.embed_image(image_path)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    e.image_key,
                    e.task_id,
                    e.image_path,
                    e.embedding_json,
                    e.created_at AS image_embedding_created_at,
                    t.session_id,
                    t.question,
                    t.task_type,
                    t.vision_answer,
                    t.critic_decision,
                    t.critic_reason,
                    t.human_decision,
                    t.human_feedback,
                    t.final_answer,
                    t.created_at AS task_created_at
                FROM image_embeddings e
                LEFT JOIN vision_tasks t
                ON e.task_id = t.task_id
                """
            )

            rows = cursor.fetchall()

        scored = []

        for row in rows:
            item = dict(row)

            if exclude_same_image and item.get("image_path") == image_path:
                continue

            try:
                doc_vector = json.loads(item["embedding_json"])
            except Exception:
                continue

            score = self.image_embedding_service.cosine_similarity(
                query_vector=query_vector,
                doc_vector=doc_vector,
            )

            if score >= min_score:
                item["image_similarity_score"] = score
                item.pop("embedding_json", None)
                scored.append(item)

        scored.sort(key=lambda x: x["image_similarity_score"], reverse=True)

        return scored[:top_k]
