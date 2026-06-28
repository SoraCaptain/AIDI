# tests/test_performance_comparison.py
import time
import asyncio
from app.tools.native_vision_tools import detect_objects
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from utils.logger import logger

async def compare_latency():
    # 1. 原生调用
    start = time.perf_counter()
    result = await detect_objects.ainvoke({"image_path": "test.jpg"})
    native_latency = time.perf_counter() - start
    logger.info(f"⏱️  原生工具延迟: {native_latency * 1000:.2f} ms")
    
    # 2. MCP 调用（需要先加载）
    client, tools = await load_vision_mcp_tools()
    mcp_detect = next(t for t in tools if t.name == "detect_objects")
    start = time.perf_counter()
    result = await mcp_detect.ainvoke({"image_path": "test.jpg"})
    mcp_latency = time.perf_counter() - start
    logger.info(f"⏱️  MCP 工具延迟: {mcp_latency * 1000:.2f} ms")
    
    logger.info(f"📊 性能提升: {(mcp_latency / native_latency - 1) * 100:.1f}%")
