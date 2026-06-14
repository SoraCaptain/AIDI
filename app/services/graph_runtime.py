# app/services/graph_runtime.py

import json
from typing import Optional, Dict, Any

from langgraph.types import Command

from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.memory.memory_manager import MemoryManager
from app.graphs.parallel_multi_agent_vision_graph import (
    build_parallel_multi_agent_vision_graph,
)
from app.observability.langfuse_client import (
    get_langfuse_handler,
    build_trace_metadata,
)


class GraphRuntime:
    """
    FastAPI 进程内的 LangGraph runtime。

    课程版：
    - 单进程
    - 内存 checkpointer
    - 内存 task store

    生产版：
    - 持久化 checkpointer
    - 外部任务队列
    - 数据库 task store
    """

    def __init__(self):
        self.app = None
        self.memory_manager = None
        self.mcp_client = None
        self.mcp_tools = None
        self.session_id = "api-gateway-session-001"

    async def initialize(self):
        self.mcp_client, self.mcp_tools = await load_vision_mcp_tools()

        self.memory_manager = MemoryManager(
            session_id=self.session_id,
            db_path="data/memory/vision_memory.sqlite3",
            max_turns=8,
            enable_vector_memory=True,
            enable_image_vector_memory=True,
        )

        self.app = build_parallel_multi_agent_vision_graph(
            mcp_tools=self.mcp_tools,
            memory_manager=self.memory_manager,
        )

    def _build_config(
        self,
        *,
        thread_id: str,
        task_id: str,
        question: str,
        image_url: Optional[str],
    ) -> Dict[str, Any]:
        langfuse_handler = get_langfuse_handler()

        trace_metadata = build_trace_metadata(
            session_id=self.session_id,
            thread_id=thread_id,
            image_path=image_url,
            question=question,
            extra={
                "task_id": task_id,
                "entrypoint": "fastapi_gateway",
                "graph": "parallel_multi_agent_vision_graph",
            },
        )

        return {
            "configurable": {
                "thread_id": thread_id,
            },
            "callbacks": [langfuse_handler],
            "metadata": trace_metadata,
            "tags": [
                "vision-agent",
                "parallel",
                "multi-agent",
                "fastapi",
                "mcp",
                "memory",
                "hitl",
            ],
        }

    async def run_task(
        self,
        *,
        task_id: str,
        thread_id: str,
        question: str,
        image_url: str,
    ) -> Dict[str, Any]:
        if self.app is None:
            raise RuntimeError("GraphRuntime is not initialized")

        # 让 MemoryManager 也知道当前图片
        self.memory_manager.set_current_image(image_url)
        self.memory_manager.add_user_message(question)

        initial_state = {
            "session_id": self.session_id,
            "question": question,
            "image_path": image_url,
            "retry_count": 0,
            "max_retries": 2,
        }

        config = self._build_config(
            thread_id=thread_id,
            task_id=task_id,
            question=question,
            image_url=image_url,
        )

        result = await self.app.ainvoke(initial_state, config=config)

        return self._normalize_result(result)

    async def resume_task(
        self,
        *,
        task_id: str,
        thread_id: str,
        question: str,
        image_url: str,
        resume_value: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.app is None:
            raise RuntimeError("GraphRuntime is not initialized")

        config = self._build_config(
            thread_id=thread_id,
            task_id=task_id,
            question=question,
            image_url=image_url,
        )

        result = await self.app.ainvoke(
            Command(resume=resume_value),
            config=config,
        )

        return self._normalize_result(result)

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        把 LangGraph result 统一转换成 API 层更容易处理的结构。
        """

        if "__interrupt__" in result:
            interrupt_obj = result["__interrupt__"][0]
            interrupt_value = interrupt_obj.value

            return {
                "status": "waiting_human",
                "interrupt": interrupt_value,
                "raw_result": result,
                "final_answer": None,
                "trace_summary": {
                    "required_agents": result.get("required_agents"),
                    "critic_decision": result.get("critic_decision"),
                    "critic_reason": result.get("critic_reason"),
                    "retry_count": result.get("retry_count"),
                },
            }

        final_answer = result.get("final_answer")

        if final_answer:
            self.memory_manager.add_assistant_message(final_answer)

        return {
            "status": "completed",
            "interrupt": None,
            "raw_result": result,
            "final_answer": final_answer,
            "trace_summary": {
                "required_agents": result.get("required_agents"),
                "critic_decision": result.get("critic_decision"),
                "critic_reason": result.get("critic_reason"),
                "human_decision": result.get("human_decision"),
                "retry_count": result.get("retry_count"),
                "task_id": result.get("task_id"),
            },
        }


graph_runtime = GraphRuntime()
