# app/graphs/plan_schema.py (新建)
from typing import List, Optional, Literal
from pydantic import BaseModel

class AgentTask(BaseModel):
    agent_name: Literal["ocr", "detection", "segmentation", "grounding_dino", "vlm_understanding", "quality"]
    depends_on: List[str] = []  # 依赖的前置任务列表，如 detection 依赖 grounding_dino
    fallback_agent: Optional[str] = None  # 如果失败，降级到哪个 Agent

class ExecutionPlan(BaseModel):
    """智能体执行计划"""
    reasoning: str = "思考过程：为什么选择这些 Agent"
    tasks: List[AgentTask] = []  # 任务列表，按顺序执行（但可配置并行）
    parallel_groups: List[List[str]] = []  # 可并行执行的任务组
    required_capabilities: List[str] = []  # 需要的服务器能力
