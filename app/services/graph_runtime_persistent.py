# app/services/graph_runtime_persistent.py
import os
from typing import Optional, Dict, Any

from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.memory.memory_manager import MemoryManager
from app.graphs.parallel_multi_agent_vision_graph import (
    build_parallel_multi_agent_vision_graph,
)
from app.observability.langfuse_client import (
    get_langfuse_handler,
    build_trace_metadata,
)
# ⬇️ 新增导入配置单例
from app.config import settings as app_settings


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

        self.mcp_client, self.mcp_tools = await load_vision_mcp_tools()

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
                print(f"🐘 正在连接 PostgreSQL: {self.postgres_uri.split('@')[-1]}")
                self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(
                    self.postgres_uri
                )
                self.checkpointer = await self._checkpointer_cm.__aenter__()
                await self.checkpointer.setup()
                print("✅ PostgreSQL 检查点已就绪")
            except Exception as e:
                print(f"⚠️  PostgreSQL 连接失败: {e}")
                print("🔄 自动降级到 SQLite")
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
            print(f"✅ SQLite 检查点已就绪: {self.checkpoint_db_path}")

        self.app = build_parallel_multi_agent_vision_graph(
            mcp_tools=self.mcp_tools,
            memory_manager=self.memory_manager,
            checkpointer=self.checkpointer,
        )

    # close, run_task, resume_task, _normalize_result 保持原样，省略复制以避免冗余...
    # （直接沿用你之前的代码即可）