# app/api/task_store_sqlite.py

import os
import json
import time
from typing import Optional, Dict, Any, List

import aiosqlite


class SQLiteTaskStore:
    """
    SQLite 持久化任务存储。

    保存：
    - task_id
    - thread_id
    - session_id
    - question
    - image_path
    - image_url
    - status
    - interrupt payload
    - final_answer
    - error
    - trace_summary
    - created_at / updated_at

    课程版：
    - 单机 SQLite
    - 适合 demo / 单机轻量部署

    生产版：
    - 推荐 Postgres / Redis
    """

    def __init__(self, db_path: str = "data/api/tasks.sqlite3"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    image_path TEXT,
                    image_url TEXT,
                    status TEXT NOT NULL,
                    interrupt_json TEXT,
                    result_json TEXT,
                    final_answer TEXT,
                    error TEXT,
                    trace_summary_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status)
                """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_thread_id
                ON tasks(thread_id)
                """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_created_at
                ON tasks(created_at)
                """
            )

            await db.commit()

    async def create_task(
        self,
        *,
        task_id: str,
        thread_id: str,
        question: str,
        image_path: str,
        image_url: str,
        session_id: str,
    ):
        now = time.time()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO tasks (
                    task_id,
                    thread_id,
                    session_id,
                    question,
                    image_path,
                    image_url,
                    status,
                    interrupt_json,
                    result_json,
                    final_answer,
                    error,
                    trace_summary_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    thread_id,
                    session_id,
                    question,
                    image_path,
                    image_url,
                    "queued",
                    None,
                    None,
                    None,
                    None,
                    None,
                    now,
                    now,
                ),
            )

            await db.commit()

    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                """
                SELECT *
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,),
            )

            row = await cursor.fetchone()

        if row is None:
            return None

        return self._decode_row(dict(row))

    async def list_tasks(
        self,
        *,
        limit: int = 50,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if status:
                cursor = await db.execute(
                    """
                    SELECT *
                    FROM tasks
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (status, limit),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT *
                    FROM tasks
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            rows = await cursor.fetchall()

        return [self._decode_row(dict(row)) for row in rows]

    async def update(self, task_id: str, **kwargs):
        allowed = {
            "status",
            "interrupt",
            "result",
            "final_answer",
            "error",
            "trace_summary",
        }

        updates = {}
        for key, value in kwargs.items():
            if key not in allowed:
                continue

            if key == "interrupt":
                updates["interrupt_json"] = json.dumps(value, ensure_ascii=False) if value is not None else None
            elif key == "result":
                updates["result_json"] = json.dumps(value, ensure_ascii=False, default=str) if value is not None else None
            elif key == "trace_summary":
                updates["trace_summary_json"] = json.dumps(value, ensure_ascii=False, default=str) if value is not None else None
            else:
                updates[key] = value

        if not updates:
            return

        updates["updated_at"] = time.time()

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.append(task_id)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"""
                UPDATE tasks
                SET {set_clause}
                WHERE task_id = ?
                """,
                values,
            )

            await db.commit()

    async def mark_unfinished_after_restart(self):
        """
        服务重启恢复策略。

        queued/running 状态说明服务可能在处理中被重启。
        课程版先标记为 interrupted_by_restart，避免误以为还在运行。

        waiting_human 不改，因为它可以通过 human-review 恢复。
        completed/failed 不改。
        """

        now = time.time()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE tasks
                SET status = ?,
                    error = ?,
                    updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (
                    "interrupted_by_restart",
                    "Service restarted while task was queued or running. Please resubmit or implement worker requeue.",
                    now,
                ),
            )

            await db.commit()

    def _decode_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["interrupt"] = self._loads(row.pop("interrupt_json", None))
        row["result"] = self._loads(row.pop("result_json", None))
        row["trace_summary"] = self._loads(row.pop("trace_summary_json", None))
        return row

    def _loads(self, value: Optional[str]):
        if not value:
            return None

        try:
            return json.loads(value)
        except Exception:
            return value


task_store = SQLiteTaskStore()
