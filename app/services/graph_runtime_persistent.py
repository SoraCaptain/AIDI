# app/services/graph_runtime_persistent.py
import os
from typing import Optional, Dict, Any

from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.tools.tool_registry import ToolRegistry
from app.memory.memory_manager import MemoryManager
from app.graphs.dynamic_vision_graph import build_dynamic_vision_graph
# from app.graphs.parallel_multi_agent_vision_graph import build_parallel_multi_agent_vision_graph
from app.agents.local_models import get_text_llm
from app.observability.langfuse_client import (
    get_langfuse_handler,
    build_trace_metadata,
)
# ⬇️ 新增导入配置单例
from app.config import settings as app_settings
from utils.logger import logger


class PersistentGraphRuntime:
    """
    持久化 LangGraph runtime。

    配置优先级：
    1. __init__ 传入的参数（显式指定）
    2. .env 或系统环境变量（通过 config.py）
    3. config.py 中的硬编码默认值
    """

    def __init__(
        self,
        checkpoint_db_path: Optional[str] = None,
        memory_db_path: Optional[str] = None,
        session_id: Optional[str] = None,
        # 检查点后端配置（如果不传，则从 config.py 读取）
        use_postgres: Optional[bool] = None,
        postgres_uri: Optional[str] = None,
    ):
        # 1. 优先使用传入参数，否则从 config.settings 读取
        self.checkpoint_db_path = checkpoint_db_path or app_settings.sqlite_checkpoint_path
        self.memory_db_path = memory_db_path or app_settings.memory_db_path
        self.session_id = session_id or app_settings.session_id
        
        self.use_postgres = (
            use_postgres
            if use_postgres is not None
            else app_settings.use_postgres_checkpointer
        )
        self.postgres_uri = (
            postgres_uri
            if postgres_uri is not None
            else app_settings.postgres_checkpointer_uri
        )

        self.app = None
        self.memory_manager = None
        self.mcp_client = None
        self.mcp_tools = None

        self._checkpointer_cm = None
        self.checkpointer = None

    # -------------------- 下面的 initialize, close, run_task 等完全保持不变 --------------------
    async def initialize(self):
        os.makedirs(os.path.dirname(self.checkpoint_db_path), exist_ok=True)

        # ---------- 🆕 使用 ToolRegistry 替代直接加载 MCP ----------
        self.tool_registry = ToolRegistry()  # 自动从 settings 读取模式
        await self.tool_registry.initialize()
        
        # 获取合并后的工具列表
        all_tools = self.tool_registry.get_tools()
        logger.info(f"🔧 最终可用工具: {len(all_tools)} 个 (模式: {self.tool_registry.get_mode()})")

        # 为了保持兼容性，仍然保留 mcp_client 引用（如果存在）
        self.mcp_client = self.tool_registry._mcp_client
        self.mcp_tools = all_tools  # 现在包含 MCP + 原生（根据模式）

        self.memory_manager = MemoryManager(
            session_id=self.session_id,
            db_path=self.memory_db_path,
            max_turns=8,
            enable_vector_memory=True,
            enable_image_vector_memory=True,
        )

        # ---------- 检查点后端选择 ----------
        if self.use_postgres:
            try:
                logger.info(f"🐘 正在连接 PostgreSQL: {self.postgres_uri.split('@')[-1]}")
                self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(
                    self.postgres_uri
                )
                self.checkpointer = await self._checkpointer_cm.__aenter__()
                await self.checkpointer.setup()
                logger.info("✅ PostgreSQL 检查点已就绪")
            except Exception as e:
                logger.warning(f"⚠️  PostgreSQL 连接失败: {e}")
                logger.info("🔄 自动降级到 SQLite")
                self.use_postgres = False
                self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(
                    self.checkpoint_db_path
                )
                self.checkpointer = await self._checkpointer_cm.__aenter__()
        else:
            self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(
                self.checkpoint_db_path
            )
            self.checkpointer = await self._checkpointer_cm.__aenter__()
            logger.info(f"✅ SQLite 检查点已就绪: {self.checkpoint_db_path}")

        llm = get_text_llm(temperature=0)

        self.app = build_dynamic_vision_graph(
            checkpointer=self.checkpointer,
            llm=llm,
            tool_registry=self.tool_registry,
        )
        # self.app = build_parallel_multi_agent_vision_graph(
        #     mcp_tools=self.mcp_tools,  # 这里传的是所有工具
        #     memory_manager=self.memory_manager,
        #     checkpointer=self.checkpointer,
        # )

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
                "graph": "dynamic_vision_graph",
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
                f"tool-mode-{self.tool_registry.get_mode() if hasattr(self, 'tool_registry') else 'unknown'}",
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
