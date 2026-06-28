# app/observability/langfuse_client.py

import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from utils.logger import logger

load_dotenv()


def get_langfuse_client():
    """
    返回 Langfuse client。
    会自动读取：
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST
    """
    return get_client()


def get_langfuse_handler() -> CallbackHandler:
    """
    LangChain / LangGraph callback handler。
    """
    return CallbackHandler()


def build_trace_metadata(
    *,
    session_id: str,
    thread_id: str,
    image_path: Optional[str],
    question: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = {
        "session_id": session_id,
        "thread_id": thread_id,
        "image_path": image_path,
        "question": question,
        "app": "visual-inspection-agent",
        "course_lesson": "lesson-10-langfuse",
    }

    if extra:
        metadata.update(extra)

    return metadata


def flush_langfuse():
    """
    CLI / short-lived process 结束前建议 flush。
    Langfuse 文档示例也建议短生命周期应用调用 flush。
    """
    client = get_langfuse_client()
    client.flush()


def score_trace_safe(
    *,
    trace_id: Optional[str],
    name: str,
    value: float,
    comment: Optional[str] = None,
):
    """
    兼容式 score 封装。
    注意：不同 Langfuse SDK 版本 score API 可能有差异。
    如果 trace_id 暂时拿不到，可以先跳过。
    """
    if not trace_id:
        return

    client = get_langfuse_client()

    try:
        client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )
    except Exception as e:
        logger.warning(f"Langfuse score failed: {repr(e)}")


# 观测清单
# 1. 是否能看到一条完整 trace？
# 2. trace metadata 里有没有 session_id / thread_id / image_path？
# 3. 是否能看到 planner LLM input/output？
# 4. 是否能看到 vision_agent？
# 5. 是否能看到 MCP tool call？
# 6. 工具 output 里有没有 latency_ms？
# 7. 是否能看到 critic decision？
# 8. human_review 是否出现 interrupt payload？
# 9. report 是否生成最终输出？
# 10. save_memory 是否写入 task_id？
