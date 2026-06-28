# tests/test_tool_registry.py
import pytest
from app.tools.tool_registry import ToolRegistry
from app.tools.native_vision_tools import NATIVE_TOOL_NAMES
from utils.logger import logger


@pytest.mark.asyncio
async def test_native_only_mode():
    registry = ToolRegistry(mode="native_only")
    await registry.initialize()
    tools = registry.get_tools()
    
    # 原生工具数量
    assert len(tools) == len(NATIVE_TOOL_NAMES)
    # 检查工具名称
    tool_names = {t.name for t in tools}
    assert "detect_blur" in tool_names
    assert "vlm_understand_image" in tool_names
    logger.info(f"✅ native_only 模式加载了 {len(tools)} 个工具")
    await registry.close()


@pytest.mark.asyncio
async def test_hybrid_mode():
    registry = ToolRegistry(mode="hybrid")
    await registry.initialize()
    tools = registry.get_tools()
    
    # hybrid 模式至少包含原生工具
    tool_names = {t.name for t in tools}
    assert "detect_blur" in tool_names
    # 如果有 MCP 服务运行，这里会更多
    logger.info(f"✅ hybrid 模式加载了 {len(tools)} 个工具")
    await registry.close()
