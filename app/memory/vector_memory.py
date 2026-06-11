# app/memory/vector_memory.py

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.memory.embedding_service import EmbeddingService


class VectorMemory:
    """
    SQLite-based vector memory.

    保存每个 task 的文本 embedding：
    - question
    - vision_answer
    - human_feedback
    - final_answer
    """

    def __init__(
        self,
        db_path: str = "data/memory/vision_memory.sqlite3",
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.embedding_service = embedding_service or EmbeddingService()

        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS task_embeddings (
                    task_id TEXT PRIMARY KEY,
                    embedding_text TEXT,
                    embedding_json TEXT,
                    created_at TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_embeddings_created_at
                ON task_embeddings(created_at)
                """
            )

            conn.commit()

    def build_embedding_text(self, task: Dict[str, Any]) -> str:
        """
        把结构化任务记录转成可 embedding 的文本。

        注意：
        不要把无关 metadata 塞太多；
        人工反馈很重要，要保留。
        """

        parts = []

        if task.get("question"):
            parts.append(f"用户问题：{task.get('question')}")

        if task.get("task_type"):
            parts.append(f"任务类型：{task.get('task_type')}")

        if task.get("vision_answer"):
            parts.append(f"视觉分析结果：{task.get('vision_answer')}")

        if task.get("critic_reason"):
            parts.append(f"审核理由：{task.get('critic_reason')}")

        if task.get("human_feedback"):
            parts.append(f"人工反馈：{task.get('human_feedback')}")

        if task.get("final_answer"):
            parts.append(f"最终报告：{task.get('final_answer')}")

        return "\n".join(parts)

    def upsert_task_embedding(self, task_id: str, task: Dict[str, Any]):
        embedding_text = self.build_embedding_text(task)

        if not embedding_text.strip():
            return

        embedding = self.embedding_service.embed_text(embedding_text)
        created_at = datetime.utcnow().isoformat()

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO task_embeddings (
                    task_id,
                    embedding_text,
                    embedding_json,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    task_id,
                    embedding_text,
                    json.dumps(embedding),
                    created_at,
                ),
            )

            conn.commit()

    def search_similar_tasks(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        基于 query 的 embedding 找语义相似任务。
        """

        if not query.strip():
            return []

        query_vector = self.embedding_service.embed_text(query)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    e.task_id,
                    e.embedding_text,
                    e.embedding_json,
                    t.session_id,
                    t.image_path,
                    t.question,
                    t.task_type,
                    t.vision_answer,
                    t.critic_decision,
                    t.critic_reason,
                    t.human_decision,
                    t.human_feedback,
                    t.final_answer,
                    t.created_at
                FROM task_embeddings e
                LEFT JOIN vision_tasks t
                ON e.task_id = t.task_id
                """
            )

            rows = cursor.fetchall()

        scored = []

        for row in rows:
            item = dict(row)

            try:
                doc_vector = json.loads(item["embedding_json"])
            except Exception:
                continue

            score = self.embedding_service.cosine_similarity(
                query_vector=query_vector,
                doc_vector=doc_vector,
            )

            if score >= min_score:
                item["similarity_score"] = score
                item.pop("embedding_json", None)
                scored.append(item)

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)

        return scored[:top_k]
