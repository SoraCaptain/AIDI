# app/memory/short_memory.py

from typing import List, Dict, Optional


class ShortMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: List[Dict] = []
        self.current_image_path: Optional[str] = None

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.history = self.history[-self.max_turns:]

    def set_image(self, image_path: str):
        self.current_image_path = image_path

    def get_image(self):
        return self.current_image_path

    def get_messages(self):
        return list(self.history)

    def clear(self):
        self.history = []
        self.current_image_path = None
