# app/skills/base.py
"""
技能系统基类
定义了一个复合技能的标准接口
"""
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.runnables import RunnableConfig
from app.observability.metrics import record_skill_call


class SkillInput(BaseModel):
    """技能输入的基类，子类可继承扩展"""
    class ConfigDict:
        arbitrary_types_allowed = True


class SkillOutput(BaseModel):
    """技能输出的标准格式"""
    success: bool = Field(description="执行是否成功")
    result: Any = Field(description="技能执行返回的数据")
    error: Optional[str] = Field(default=None, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class BaseSkill(ABC):
    """
    技能抽象基类
    每个技能封装了一个完整的业务工作流
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """技能名称（唯一标识）"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """技能描述（用于 LLM 决策）"""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseModel]:
        """输入参数的 Pydantic Schema"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> SkillOutput:
        """
        执行技能的核心逻辑
        可以在这里编排多个原子工具调用
        """
        pass

    def to_langchain_tool(self) -> BaseTool:
        """将 Skill 转换为 LangChain 的 BaseTool，以便 Agent 直接调用"""
        # 构造一个异步函数，调用 self.execute
        async def _run(**kwargs):
            result = await self.execute(**kwargs)
            # 如果出错，抛出异常让 Agent 捕获
            if not result.success:
                raise RuntimeError(result.error or "Skill execution failed")
            return result.result

        return StructuredTool.from_function(
            name=self.name,
            description=self.description,
            func=_run,
            args_schema=self.input_schema,
        )

    async def execute(self, **kwargs):
        start = time.time()
        try:
            result = await self._execute_impl(**kwargs)
            status = "success"
            return result
        except Exception:
            status = "failure"
            raise
        finally:
            record_skill_call(self.name, status)
