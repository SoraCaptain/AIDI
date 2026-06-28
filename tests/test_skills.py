# tests/test_skills.py
import pytest
import asyncio
from app.skills.builtin.vision_composer import ComprehensiveAnalysisSkill
from app.skills.registry import SkillRegistry
from utils.logger import logger

@pytest.mark.asyncio
async def test_comprehensive_skill():
    skill = ComprehensiveAnalysisSkill()
    
    # 注意：请替换为实际存在的测试图片路径
    result = await skill.execute(
        image_path="test_imgs/table_chair.jpg",
        question="分析这个场景的布局和主要物体"
    )
    
    assert result.success is True
    assert "report" in result.result
    logger.info(result.result["report"])  # 打印生成的 Markdown 报告

@pytest.mark.asyncio
async def test_skill_registry_tools():
    registry = SkillRegistry()
    tools = registry.get_tools()
    
    assert len(tools) >= 1
    tool_names = [t.name for t in tools]
    assert "comprehensive_image_analysis" in tool_names
    logger.info(f"✅ 技能工具列表: {tool_names}")
