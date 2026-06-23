# app/skills/md_skill_adapter.py
"""
将 SKILL.md 技能适配为 LangChain 工具
当 Agent 调用该工具时，会将完整的 SKILL.md 内容注入上下文，引导 LLM 按步骤执行。
"""
import json
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.runnables import RunnableConfig

from app.skills.skill_loader import SkillLoader


class MDSkillInput(BaseModel):
    """MD 技能的标准输入"""
    instruction: str = Field(description="用户的具体指令或问题，技能将根据此指令执行")
    additional_context: Optional[str] = Field(default="", description="额外的上下文信息")


class MDSkillTool(BaseTool):
    """
    将 SKILL.md 封装成一个工具：
    当调用时，返回完整的技能文档（作为系统指令），
    并让调用者（Agent）自行按照文档执行。
    """
    name: str
    description: str
    skill_content: str  # 完整的 SKILL.md 内容

    def _run(self, **kwargs) -> str:
        # 同步模式（不常用）
        raise NotImplementedError("MD Skill 仅支持异步调用")

    async def _arun(self, instruction: str, additional_context: str = "", **kwargs) -> str:
        """
        异步执行：返回格式化的指令，让 Agent 根据 SKILL.md 完成工作
        实际执行由 Agent 的后续推理完成，这里只提供指导性内容
        """
        # 将用户指令和技能文档组合成一个提示
        prompt = f"""
            # 请按照以下技能指导完成用户任务

            ## 技能文档 (来自 SKILL.md)
            {self.skill_content}

            ## 用户指令
            {instruction}

            {additional_context}

            请严格按照技能文档中的步骤执行，并输出最终结果。
            """
        # 这里不真正执行，而是返回一个“思维链”提示，由上层 LLM 处理
        # 但在工具调用场景中，我们希望这个工具是一个“元指令”，
        # 更好的做法是返回 prompt，然后由 System Message 处理。
        # 此处我们返回提示内容，并由 Agent 将其作为部分消息。
        return prompt


def create_md_skill_tool(skill_name: str, loader: SkillLoader) -> Optional[BaseTool]:
    """
    根据技能名称，创建一个 LangChain 工具
    """
    info = loader.get_skill_info(skill_name)
    if not info:
        return None

    full_content = loader.load_full_skill(skill_name)
    if not full_content:
        return None

    # 提取 frontmatter 中的描述（优先），否则使用索引中的描述
    desc = info.get("description", "执行基于 Markdown 文档定义的技能")

    return MDSkillTool(
        name=skill_name,
        description=desc,
        skill_content=full_content,
        args_schema=MDSkillInput,
    )


def load_all_md_skills(loader: SkillLoader) -> list[BaseTool]:
    """
    加载目录下所有 SKILL.md 技能，返回 LangChain 工具列表
    """
    tools = []
    for name in loader.list_skill_names():
        tool = create_md_skill_tool(name, loader)
        if tool:
            tools.append(tool)
    return tools
