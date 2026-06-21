import pytest
import asyncio
from app.services.graph_runtime_persistent import PersistentGraphRuntime

@pytest.mark.asyncio
async def test_postgres_connection():
    # 强制启用 PostgreSQL（如果环境变量没设，这里显式指定）
    runtime = PersistentGraphRuntime(use_postgres=True)
    await runtime.initialize()
    assert runtime.use_postgres is True
    assert runtime.checkpointer is not None
    await runtime.close()

@pytest.mark.asyncio
async def test_fallback_to_sqlite():
    # 用一个错误的 URI 测试降级
    runtime = PersistentGraphRuntime(
        use_postgres=True,
        postgres_uri="postgres://wrong:wrong@localhost:9999/bad"
    )
    await runtime.initialize()
    # 降级后 use_postgres 会被改为 False
    assert runtime.use_postgres is False
    await runtime.close()
