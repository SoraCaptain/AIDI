# app/observability/langfuse_client.py

import os
import time
from typing import Dict, Any, Optional
from uuid import UUID

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler
from contextlib import asynccontextmanager
from utils.logger import logger

load_dotenv()

# 模块级变量用于跨 LangGraph context 传递当前 trace 上下文
# ContextVar 在 LangChain callback 隔离的 context 中不可用，
# 所以使用普通的模块级变量（本应用为单请求模式）
_current_trace_id: Optional[str] = None
_current_observation_id: Optional[str] = None


class _TracingCallbackHandler(_LangfuseCallbackHandler):
    """自定义 CallbackHandler，在 on_chain_start 时将 trace/observation id 写入模块级变量。"""

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        result = super().on_chain_start(
            serialized, inputs,
            run_id=run_id, parent_run_id=parent_run_id,
            tags=tags, metadata=metadata, **kwargs,
        )
        if self.last_trace_id:
            global _current_trace_id, _current_observation_id
            _current_trace_id = self.last_trace_id
            obs = self._runs.get(run_id)
            if obs is not None:
                _current_observation_id = str(obs.id)
        return result

    def on_chain_end(self, outputs, *, run_id: UUID, parent_run_id=None, **kwargs):
        result = super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        global _current_observation_id
        parent_obs = self._runs.get(parent_run_id) if parent_run_id else None
        if parent_obs is not None:
            _current_observation_id = str(parent_obs.id)
        return result


def get_langfuse_client():
    """
    返回 Langfuse client。
    会自动读取：
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST
    """
    return get_client()


def get_langfuse_handler() -> _TracingCallbackHandler:
    """
    LangChain / LangGraph callback handler。
    使用自定义子类，通过模块级变量在 LangGraph node 间传递 trace 上下文。
    会覆盖父类的 on_chain_start / on_chain_end 以实现 trace 信息传递。
    """
    return _TracingCallbackHandler()


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


@asynccontextmanager
async def trace_span(name: str, metadata: Optional[Dict] = None):
    """
    手动创建 Langfuse span，自动继承当前 LangGraph trace 上下文。

    trace_id 和 observation_id 通过模块级变量从 _TracingCallbackHandler 传递，
    绕过 LangChain callback context 隔离的限制。
    未配置 LANGFUSE_PUBLIC_KEY 时静默跳过。
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        yield None
        return

    trace_id = _current_trace_id
    obs_id = _current_observation_id

    span_kwargs: Dict[str, Any] = {"name": name}
    if metadata:
        span_kwargs["metadata"] = metadata
    if trace_id:
        tc: Dict[str, str] = {"trace_id": trace_id}
        if obs_id:
            tc["parent_span_id"] = obs_id
        span_kwargs["trace_context"] = tc

    client = get_langfuse_client()
    start = time.time()
    try:
        with client.start_as_current_observation(**span_kwargs) as current_span:
            yield current_span
            duration_s = time.time() - start
            try:
                current_span.update(metadata={"duration_s": round(duration_s, 3)})
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Langfuse trace_span '{name}' failed: {repr(e)}")
        yield None

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
