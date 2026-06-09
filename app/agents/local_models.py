# app/agents/local_models.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_text_llm(temperature: float = 0):
    """
    返回连接 A800 本地 vLLM 的文本 LLM。
    通过 OpenAI-compatible API 调用。
    """

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
    )
