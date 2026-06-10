import re
from typing import Optional, Dict, List

from app.memory.session_memory import SessionMemory
from app.memory.persistent_memory import PersistentMemory


class MemoryManager:
    """
    统一记忆管理器。

    负责：
    - 当前会话短期记忆
    - SQLite 长期记忆
    - 为 LangGraph 提供 memory_context
    - 保存最终任务结果
    """

    def __init__(
        self,
        session_id: str,
        db_path: str = "data/memory/vision_memory.sqlite3",
        max_turns: int = 8,
    ):
        self.session_id = session_id
        self.session = SessionMemory(max_turns=max_turns)
        self.persistent = PersistentMemory(db_path=db_path)

    def set_current_image(self, image_path: str):
        self.session.set_image(image_path)

    def get_current_image(self) -> Optional[str]:
        return self.session.get_image()

    def add_user_message(self, content: str):
        self.session.add_message("user", content)

    def add_assistant_message(self, content: str):
        self.session.add_message("assistant", content)
        self.session.set_last_result(content)

    def get_conversation_history(self) -> List:
        return self.session.get_messages()

    def get_last_result(self) -> Optional[str]:
        return self.session.get_last_result()

    def build_memory_context(
        self,
        question: str,
        image_path: Optional[str],
    ) -> Dict:
        """
        给 LangGraph load_memory 节点使用。
        """

        recent_tasks = self.persistent.get_recent_tasks(
            session_id=self.session_id,
            limit=3,
        )

        same_image_tasks = []
        if image_path:
            same_image_tasks = self.persistent.get_tasks_by_image(
                image_path=image_path,
                limit=3,
            )

        keywords = self._extract_keywords(question)
        keyword_tasks = []

        for kw in keywords[:3]:
            keyword_tasks.extend(
                self.persistent.search_keyword(
                    keyword=kw,
                    limit=2,
                )
            )

        keyword_tasks = self._deduplicate_tasks(keyword_tasks)

        return {
            "session_id": self.session_id,
            "conversation_history": self.get_conversation_history(),
            "last_result": self.get_last_result(),
            "recent_tasks": recent_tasks,
            "same_image_tasks": same_image_tasks,
            "keyword_tasks": keyword_tasks,
        }

    def save_graph_result(self, state: Dict) -> str:
        task_id = self.persistent.save_task(
            {
                "session_id": self.session_id,
                "image_path": state.get("image_path"),
                "question": state.get("question"),
                "task_type": state.get("task_type"),
                "planner_reason": state.get("planner_reason"),
                "vision_answer": state.get("vision_answer"),
                "critic_decision": state.get("critic_decision"),
                "critic_reason": state.get("critic_reason"),
                "human_decision": state.get("human_decision"),
                "human_feedback": state.get("human_feedback"),
                "final_answer": state.get("final_answer"),
            }
        )

        return task_id

    def _extract_keywords(self, text: str) -> List:
        """
        简单关键词抽取。
        暂时用规则，后面可以替换成 LLM 或 embedding。
        """

        if not text:
            return []

        candidates = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9_]{2,}", text)

        stopwords = {
            "这张图",
            "图片",
            "是否",
            "有没有",
            "请帮我",
            "分析",
            "问题",
            "什么",
            "一个",
            "这个",
        }

        keywords = []
        for item in candidates:
            if item not in stopwords and item not in keywords:
                keywords.append(item)

        return keywords

    def _deduplicate_tasks(self, tasks: List[Dict]) -> List:
        seen = set()
        result = []

        for task in tasks:
            task_id = task.get("task_id")
            if task_id in seen:
                continue
            seen.add(task_id)
            result.append(task)

        return result