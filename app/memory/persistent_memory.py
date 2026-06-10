import os
import sqlite3
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any


class PersistentMemory:
    """
    SQLite 长期记忆。

    用于保存历史视觉分析任务、报告、人工反馈。
    """

    def __init__(self, db_path: str = "data/memory/vision_memory.sqlite3"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    image_path TEXT,
                    question TEXT,
                    task_type TEXT,
                    planner_reason TEXT,
                    vision_answer TEXT,
                    critic_decision TEXT,
                    critic_reason TEXT,
                    human_decision TEXT,
                    human_feedback TEXT,
                    final_answer TEXT,
                    created_at TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vision_tasks_session_id
                ON vision_tasks(session_id)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vision_tasks_image_path
                ON vision_tasks(image_path)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vision_tasks_created_at
                ON vision_tasks(created_at)
                """
            )

            conn.commit()

    def save_task(self, data: Dict[str, Any]) -> str:
        task_id = data.get("task_id") or str(uuid.uuid4())
        created_at = data.get("created_at") or datetime.utcnow().isoformat()

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO vision_tasks (
                    task_id,
                    session_id,
                    image_path,
                    question,
                    task_type,
                    planner_reason,
                    vision_answer,
                    critic_decision,
                    critic_reason,
                    human_decision,
                    human_feedback,
                    final_answer,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    data.get("session_id"),
                    data.get("image_path"),
                    data.get("question"),
                    data.get("task_type"),
                    data.get("planner_reason"),
                    data.get("vision_answer"),
                    data.get("critic_decision"),
                    data.get("critic_reason"),
                    data.get("human_decision"),
                    data.get("human_feedback"),
                    data.get("final_answer"),
                    created_at,
                ),
            )

            conn.commit()

        return task_id

    def get_recent_tasks(
        self,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> List:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if session_id:
                cursor.execute(
                    """
                    SELECT *
                    FROM vision_tasks
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM vision_tasks
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_tasks_by_image(
        self,
        image_path: str,
        limit: int = 5,
    ) -> List:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT *
                FROM vision_tasks
                WHERE image_path = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (image_path, limit),
            )

            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def search_keyword(
        self,
        keyword: str,
        limit: int = 5,
    ) -> List:
        """
        简单关键词检索。
        下一课会升级为向量检索。
        """

        pattern = f"%{keyword}%"

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT *
                FROM vision_tasks
                WHERE
                    question LIKE ?
                    OR vision_answer LIKE ?
                    OR final_answer LIKE ?
                    OR human_feedback LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, pattern, limit),
            )

            rows = cursor.fetchall()

        return [dict(row) for row in rows]
