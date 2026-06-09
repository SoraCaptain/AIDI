# app/memory/session_memory.py

from typing import Optional, List, Dict


class SessionMemory:
    """
    简单会话记忆。
    用于保存：
    - 当前图片
    - 最近对话
    - 最近一次分析结果
    """

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.current_image_path: Optional[str] = None
        self.messages: List[Dict] = []
        self.last_result: Optional[str] = None

    def set_image(self, image_path: str):
        self.current_image_path = image_path

    def get_image(self) -> Optional[str]:
        return self.current_image_path

    def add_message(self, role: str, content: str):
        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )
        self.messages = self.messages[-self.max_turns:]

    def get_messages(self) -> List[Dict]:
        return self.messages

    def set_last_result(self, result: str):
        self.last_result = result

    def get_last_result(self) -> Optional[str]:
        return self.last_result

    def clear(self):
        self.current_image_path = None
        self.messages = []
        self.last_result = None
