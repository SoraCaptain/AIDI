## 这个模块负责根据模式加载 MCP 工具或原生工具，并统一返回 LangChain BaseTool 列表。


# app/tools/tool_registry.py
"""
统一工具注册中心
支持 mcp_only / native_only / hybrid 三种模式
"""
import os
from typing import List, Optional, Tuple
from langchain_core.tools import BaseTool

from app.config import settings
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.tools.native_vision_tools import NATIVE_VISION_TOOLS
from app.skills.registry import SkillRegistry
from app.skills.skill_loader import SkillLoader
from app.skills.md_skill_adapter import load_all_md_skills
from utils.logger import logger



class ToolRegistry:
    """工具注册中心，管理 MCP 工具和原生工具的加载与合并"""

    def __init__(self, mode: Optional[str] = None):
        """
        Args:
            mode: "mcp_only" | "native_only" | "hybrid"
                  默认从 settings.tool_execution_mode 读取
        """
        self.mode = mode or settings.tool_execution_mode
        self._mcp_tools: List[BaseTool] = []
        self._native_tools: List[BaseTool] = NATIVE_VISION_TOOLS
        self._mcp_client = None
        self._is_initialized = False

    async def initialize(self):
        """加载 MCP 工具（如果需要的话）"""
        if self._is_initialized:
            return

        # 仅在需要 MCP 的模式下加载
        if self.mode in ("mcp_only", "hybrid"):
            try:
                client, tools = await load_vision_mcp_tools()
                self._mcp_client = client
                self._mcp_tools = tools
                logger.info(f"✅ MCP 工具加载完成: {len(tools)} 个")
            except Exception as e:
                logger.warning(f"⚠️  MCP 工具加载失败: {e}")
                if self.mode == "mcp_only":
                    # 如果强制 MCP 但连不上，必须抛出异常
                    raise RuntimeError(f"MCP 模式加载失败: {e}")
                else:
                    # hybrid 模式下可降级为 native_only
                    logger.info("🔄 降级为 native_only 模式")
                    self.mode = "native_only"

        # 加载复合技能（将其转换为工具）
        if settings.enable_skills:
            skill_registry = SkillRegistry()
            self._skill_tools = skill_registry.get_tools()
        else:
            self._skill_tools = []
        logger.info(f"✅ 加载了 {len(self._skill_tools)} 个复合技能")
        
        # 加载 MD 技能
        self._md_skill_tools = []
        if settings.enable_md_skills:
            loader = SkillLoader(settings.md_skills_dir)
            loader.load_index()
            self._md_skill_tools = load_all_md_skills(loader)
            logger.info(f"✅ 加载了 {len(self._md_skill_tools)} 个 MD 技能")
            # 可选：保存 loader 供后续使用
            self._md_loader = loader

        logger.info(f"✅ 原生工具加载完成: {len(self._native_tools)} 个")
        self._is_initialized = True

    def get_tools(self) -> List[BaseTool]:
        """获取所有工具（根据当前模式）"""
        if not self._is_initialized:
            raise RuntimeError("请先调用 initialize()")

        all_tools = {}

        if self.mode in ("native_only", "hybrid"):
            for tool in self._native_tools:
                all_tools[tool.name] = tool
            for tool in self._md_skill_tools:
                if tool.name not in all_tools:
                    all_tools[tool.name] = tool
            for tool in self._skill_tools:
                all_tools[tool.name] = tool
                
        if self.mode in ("mcp_only", "hybrid"):
            for tool in self._mcp_tools:
                if tool.name not in all_tools:
                    all_tools[tool.name] = tool
        logger.info(f"加载工具：{list(all_tools.keys())}")
        return list(all_tools.values())

    async def close(self):
        return
        # """释放资源（关闭 MCP 客户端）"""
        # if self._mcp_client:
        #     try:
        #         await self._mcp_client.close()
        #     except Exception as e:
        #         print(f"⚠️  关闭 MCP 客户端时出错: {e}")

    def get_mode(self) -> str:
        return self.mode

    def get_stats(self) -> dict:
        """获取统计信息（用于可观测性）"""
        return {
            "mode": self.mode,
            "native_tools_count": len(self._native_tools),
            "mcp_tools_count": len(self._mcp_tools),
            "skill_tools_count": len(self._skill_tools),
            "total_tools": len(self.get_tools()),
        }
