# tests/check_skill_tool.py
import asyncio
from app.tools.tool_registry import ToolRegistry

async def check_skill():
    registry = ToolRegistry(mode="hybrid")
    await registry.initialize()
    
    tools = registry.get_tools()
    print(f"✅ 总共加载了 {len(tools)} 个工具")
    
    skill_tool = None
    for tool in tools:
        if tool.name == "image-analysis-report":
            skill_tool = tool
            break
    
    if skill_tool:
        print(f"✅ 找到技能工具: {skill_tool.name}")
        print(f"   描述: {skill_tool.description}")
        print(f"   参数: {skill_tool.args_schema.schema()}")
        
        # 模拟调用
        result = await skill_tool._arun(
            instruction="请分析 /tmp/test.jpg 的物体和文字"
        )
        print(f"\n📄 技能返回的指导内容（前500字符）:\n{result[:500]}...")
    else:
        print("❌ 未找到 image-analysis-report 技能，请检查:")
        print("  1. 是否在 config.py 中设置 enable_md_skills=True")
        print("  2. app/skills/md_skills/image-analysis-report/SKILL.md 是否存在")

if __name__ == "__main__":
    asyncio.run(check_skill())