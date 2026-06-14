# app/api/task_store.py

import time
from typing import Dict, Any, Optional


class InMemoryTaskStore:
    """
    课程版任务存储。

    注意：
    - 进程重启会丢
    - 多 worker 不共享
    - 生产环境要换 SQLite / Redis / Postgres
    """

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(
        self,
        *,
        task_id: str,
        thread_id: str,
        question: str,
        image_path: str,
        image_url: str,
        session_id: str,
    ):
        self.tasks[task_id] = {
            "task_id": task_id,
            "thread_id": thread_id,
            "session_id": session_id,
            "question": question,
            "image_path": image_path,
            "image_url": image_url,
            "status": "queued",
            "interrupt": None,
            "result": None,
            "final_answer": None,
            "error": None,
            "trace_summary": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.tasks.get(task_id)

    def update(self, task_id: str, **kwargs):
        task = self.tasks.get(task_id)
        if not task:
            return

        task.update(kwargs)
        task["updated_at"] = time.time()

    def list_tasks(self):
        return list(self.tasks.values())


task_store = InMemoryTaskStore()
