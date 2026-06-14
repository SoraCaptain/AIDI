# app/services/graph_runtime_persistent.py

import os
from typing import Optional, Dict, Any

from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.memory.memory_manager import MemoryManager
from app.graphs.parallel_multi_agent_vision_graph import (
    build_parallel_multi_agent_vision_graph,
)
from app.observability.langfuse_client import (
    get_langfuse_handler,
    build_trace_metadata,
)


class PersistentGraphRuntime:
    """
    持久化 LangGraph runtime。

    使用：
    - SQLite Task Store 保存 API task metadata
    - AsyncSqliteSaver 保存 LangGraph checkpoint

    关键点：
    - 同一个 thread_id 可以跨服务重启恢复
    - waiting_human 任务可以继续 Command(resume=...)
    """

    def __init__(
        self,
        checkpoint_db_path: str = "data/checkpoint/langgraph_checkpoints.sqlite3",
        memory_db_path: str = "data/memory/vision_memory.sqlite3",
    ):
        self.checkpoint_db_path = checkpoint_db_path
        self.memory_db_path = memory_db_path

        self.app = None
        self.memory_manager = None
        self.mcp_client = None
        self.mcp_tools = None

        self.session_id = "api-gateway-session-001"

        self._checkpointer_cm = None
        self.checkpointer = None

    async def initialize(self):
        os.makedirs(os.path.dirname(self.checkpoint_db_path), exist_ok=True)

        self.mcp_client, self.mcp_tools = await load_vision_mcp_tools()

        self.memory_manager = MemoryManager(
            session_id=self.session_id,
            db_path=self.memory_db_path,
            max_turns=8,
            enable_vector_memory=True,
            enable_image_vector_memory=True,
        )

        # 保持 AsyncSqliteSaver context 在 FastAPI 生命周期内打开
        self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(
            self.checkpoint_db_path
        )
        self.checkpointer = await self._checkpointer_cm.__aenter__()

        self.app = build_parallel_multi_agent_vision_graph(
            mcp_tools=self.mcp_tools,
            memory_manager=self.memory_manager,
            checkpointer=self.checkpointer,
        )

    async def close(self):
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(None, None, None)

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
                "checkpoint": "sqlite",
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
                "sqlite-checkpoint",
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
            raise RuntimeError("PersistentGraphRuntime is not initialized")

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
            raise RuntimeError("PersistentGraphRuntime is not initialized")

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


graph_runtime = PersistentGraphRuntime()
