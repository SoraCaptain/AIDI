# tests/test_md_skills.py
import pytest
import asyncio
from app.skills.skill_loader import SkillLoader
from app.skills.md_skill_adapter import create_md_skill_tool, load_all_md_skills
from app.config import settings
from utils.logger import logger


@pytest.mark.asyncio
async def test_load_md_skill():
    loader = SkillLoader(settings.md_skills_dir)
    loader.load_index()
    assert "image-analysis-report" in loader.list_skill_names()
    logger.info("✅ 技能索引加载成功")

    # 获取完整内容
    content = loader.load_full_skill("image-analysis-report")
    assert content is not None
    assert "Image Analysis Report Skill" in content
    logger.info(f"✅ 技能内容长度: {len(content)} 字符")

    # 创建工具
    tool = create_md_skill_tool("image-analysis-report", loader)
    assert tool is not None
    logger.info(f"✅ 工具名称: {tool.name}")
    logger.info(f"✅ 工具描述: {tool.description}")

    # 模拟调用（返回指导性 prompt）
    result = await tool._arun(instruction="分析图片 /path/to/img.jpg")
    assert "按照技能文档" in result
    logger.info("✅ 工具调用返回指导提示")

if __name__ == "__main__":
    asyncio.run(test_load_md_skill())