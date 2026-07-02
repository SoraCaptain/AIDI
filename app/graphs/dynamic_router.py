# app/graphs/dynamic_router.py
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from app.graphs.plan_schema import ExecutionPlan, AgentTask
from app.tools.tool_registry import ToolRegistry
from app.agents import run_ocr, run_detection, run_segmentation, run_grounding_dino, run_vlm, run_quality
from app.observability.metrics import record_agent_execution, record_plan_type
from app.observability.langfuse_client import trace_span
from utils.logger import logger


def safe_json_loads(text: str) -> dict:
    """Parse JSON from LLM output, handling markdown code blocks."""
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}

    return {}


class DynamicRouter:
    """
    动态路由器：根据用户输入生成执行计划，并调度 Agent
    """

    def __init__(self, llm, tool_registry: ToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry

    # ========== 1. 意图分析与计划生成 ==========
    async def generate_plan(self, state: Dict[str, Any]) -> ExecutionPlan:
        """
        Planner 节点：分析用户问题，生成执行计划
        """
        question = state.get("question", "")
        image_path = state.get("image_path", "")

        # ========== 1. 检查服务器健康状态 ==========
        # 假设你需要的所有能力列表，可以从配置或全量 Agent 中获取
        all_capabilities = ["cv_server", "vlm_server", "gdino_server"]
        health_status = await self.check_server_health(all_capabilities)

        # 构造健康状态描述（供 LLM 参考）
        health_desc = "\n".join([
            f"- {server}: {'✅ 可用' if healthy else '❌ 不可用'}"
            for server, healthy in health_status.items()
        ])

        # 获取当前可用工具列表（用于提示 LLM 哪些能力可用）
        available_tools = self.tool_registry.get_tools()
        tool_names = [t.name for t in available_tools]

        # 构造 Prompt（这里我们利用前面学的 SKILL.md 思想，直接写指令）
        system_prompt = f"""
            你是一个视觉任务编排专家。根据用户问题，决定调用哪些视觉 Agent。

            **当前服务器健康状态：**
            {health_desc}

            如果某个 Agent 依赖的服务器不可用，**不要**在计划中包含该 Agent，或者使用其 `fallback_agent` 替代。

            可用 Agent 能力：
            1. ocr: 提取图像中的文字 (依赖 cv_server)
            2. detection: 检测物体位置和类别 (依赖 cv_server)
            3. segmentation: 实例分割（生成像素级掩码），通常依赖 detection (依赖 cv_server)
            4. grounding_dino: 根据文本描述定位特定物体 (依赖 gdino_server)
            5. vlm_understanding: 多模态大模型理解（擅长复杂场景描述、推理）(依赖 vlm_server)
            6. quality: 检查图片质量，检查图片清晰度，提取图片基本信息 (依赖 cv_server)

            规则：
            - 如果用户问图片质量如何或想知道图片的基本信息，则调用quality
            - 如果用户只问文字，只用 ocr。
            - 如果用户问“有什么物体”，用 detection + vlm（用于描述）。
            - 如果用户问“有多少/在哪里”，用 detection。
            - 如果用户需要精确定位特定物体（如“红色的车”），用 grounding_dino。
            - 如果用户要求“像素级/详细分割”，用 segmentation（如果服务器可用）。
            - 复杂场景分析（如“描述氛围”），用 vlm_understanding。
            - 如果 detection 不可用（或过载），降级到 vlm_understanding 进行粗略描述。
            - **如果某个依赖的服务器不可用，你必须在计划中避免使用该 Agent，或使用指定的降级方案。**

            请以 JSON 格式输出执行计划。
            输出格式：
            {{
            "reasoning": "思考过程",
            "tasks": [ {{ "agent_name": "detection", "depends_on": [], "fallback_agent": "vlm_understanding" }} ],
            "parallel_groups": [ ["ocr", "detection"] ],  // 这些可并行
            "required_capabilities": ["cv_server", "vlm_server", "gdino_server"]
            }}
            """
        human_prompt = f"用户问题：{question}\n图像路径：{image_path}"

        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        # 解析 LLM 返回的 JSON
        try:
            plan_dict = safe_json_loads(response.content)
            plan = ExecutionPlan(**plan_dict)

            # 后处理：根据实际健康状态，移除不可用的 Agent（如果 LLM 没遵守规则）
            plan.tasks = [
                task for task in plan.tasks
                if self._is_agent_available(task.agent_name, health_status)
                or (task.fallback_agent and self._is_agent_available(task.fallback_agent, health_status))
            ]
            return plan
        except Exception as e:
            # 默认降级策略：全部运行（兼容旧逻辑）
            logger.warning(f"⚠️  计划解析失败，使用默认全量计划: {e}")
            # 降级计划也考虑健康状态
            default_tasks = []
            if health_status.get("cv_server", False):
                default_tasks.extend([
                    AgentTask(agent_name="ocr"),
                    AgentTask(agent_name="detection"),
                    AgentTask(agent_name="segmentation", depends_on=["detection"]),
                    AgentTask(agent_name="quality"),
                ])
            if health_status.get("gdino_server", False):
                default_tasks.append(AgentTask(agent_name="grounding_dino"))
            if health_status.get("vlm_server", False):
                default_tasks.append(AgentTask(agent_name="vlm_understanding"))

            return ExecutionPlan(
                reasoning="降级到动态全量模式（按健康状态过滤）",
                tasks=default_tasks,
                parallel_groups=[["ocr", "detection"], ["vlm_understanding"], ["quality"], ["grounding_dino"]]
            )

    def _is_agent_available(self, agent_name: str, health_status: dict) -> bool:
        """判断某个 Agent 所需的服务器是否健康"""
        server_map = {
            "ocr": "cv_server",
            "detection": "cv_server",
            "segmentation": "cv_server",
            "grounding_dino": "gdino_server",
            "vlm_understanding": "vlm_server",
            "quality": "cv_server",
        }
        required = server_map.get(agent_name)
        if not required:
            return True  # 如果没有映射，默认可用
        return health_status.get(required, False)

    # ========== 2. 服务器健康检查（负载感知） ==========
    async def check_server_health(self, capabilities: List[str]) -> Dict[str, bool]:
        """
        检查所需服务器是否健康/低负载
        这里简单做 HTTP ping，实际可以检查队列深度
        """
        health_status = {}
        # 假设你有配置
        from app.config import settings

        # 模拟健康检查（实际应发请求）
        if "cv_server" in capabilities:
            try:
                # 简单的 ping
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{settings.cv_server}/health", timeout=2.0)
                    health_status["cv_server"] = resp.status_code == 200
            except:
                health_status["cv_server"] = False

        if "gdino_server" in capabilities:
            try:
                # 简单的 ping
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{settings.gdino_server}/health", timeout=2.0)
                    health_status["gdino_server"] = resp.status_code == 200
            except:
                health_status["gdino_server"] = False

        if "vlm_server" in capabilities:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{settings.vlm_server}/health", timeout=2.0)
                    health_status["vlm_server"] = resp.status_code == 200
            except:
                health_status["vlm_server"] = False

        return health_status

    # ========== 3. 动态任务调度器 ==========
    async def execute_plan(self, state: Dict[str, Any], plan: Dict[str, Any] | ExecutionPlan) -> Dict[str, Any]:
        """
        核心调度器：根据计划执行任务，支持并行和依赖
        """
        if isinstance(plan, dict):
            plan = ExecutionPlan(**plan)

        plan_type = "_".join(sorted(t.agent_name for t in plan.tasks)) if plan.tasks else "empty"
        record_plan_type(plan_type)

        async with trace_span("execute_plan", {
            "plan_type": plan_type,
            "agent_count": len(plan.tasks) if plan.tasks else 0,
        }):
            # 3.1 检查服务器健康，过滤不可用的 Agent
            health = await self.check_server_health(plan.required_capabilities)
            available_agents = []

            for task in plan.tasks:
                # 检查依赖的服务器是否健康
                required_servers = []
                if task.agent_name in ["ocr", "detection", "segmentation"]:
                    required_servers.append("cv_server")
                if task.agent_name in ["grounding_dino"]:
                    required_servers.append("gdino_server")
                if task.agent_name in ["vlm_understanding"]:
                    required_servers.append("vlm_server")

                is_healthy = all(health.get(s, True) for s in required_servers)
                if not is_healthy and task.fallback_agent:
                    logger.warning(f"⚠️  {task.agent_name} 服务器不可用，降级到 {task.fallback_agent}")
                    fallback_task = AgentTask(
                        agent_name=task.fallback_agent,
                        depends_on=task.depends_on,
                        fallback_agent=None
                    )
                    available_agents.append(fallback_task)
                elif is_healthy:
                    available_agents.append(task)
                else:
                    logger.warning(f"❌ 跳过 {task.agent_name}（服务器不可用且无降级）")

            # 3.2 构造并行 DAG
            serial_tasks = []
            parallel_batches = []

            dependent_tasks = [t for t in available_agents if t.depends_on]
            independent_tasks = [t for t in available_agents if not t.depends_on]

            if independent_tasks:
                parallel_batches.append(independent_tasks)

            for task in dependent_tasks:
                serial_tasks.append(task)

            # 3.3 执行任务并收集结果
            results = {}

            # 执行并行批次
            for batch in parallel_batches:
                batch_results = await asyncio.gather(*[
                    self._run_single_agent(task.agent_name, state, results) for task in batch
                ])
                for task, res in zip(batch, batch_results):
                    results[task.agent_name] = res

            # 执行串行任务（等待依赖完成）
            for task in serial_tasks:
                for dep in task.depends_on:
                    if dep not in results:
                        logger.info(f"⚠️  依赖 {dep} 未执行，尝试自动补齐")
                        results[dep] = await self._run_single_agent(dep, state, results)
                results[task.agent_name] = await self._run_single_agent(
                    task.agent_name, state, results
                )

            return results

    async def _run_single_agent(self, agent_name: str, state: Dict, context_results: Dict, timeout: int = 90) -> Any:
        """
        执行单个 Agent，并注入已有的上下文结果
        """
        start = time.time()
        status = "success"
        logger.info(f"▶️  运行 Agent: {agent_name}")

        enhanced_state = {**state, "context_results": context_results}

        agent_map = {
            "ocr": run_ocr,
            "detection": run_detection,
            "segmentation": run_segmentation,
            "grounding_dino": run_grounding_dino,
            "vlm_understanding": run_vlm,
            "quality": run_quality
        }
        func = agent_map.get(agent_name)
        if not func:
            record_agent_execution(agent_name, time.time() - start, "failure")
            return {"error": f"Unknown agent: {agent_name}"}

        async with trace_span(f"agent.{agent_name}", {"timeout": timeout}):
            try:
                result = await asyncio.wait_for(func(enhanced_state), timeout=timeout)
                if isinstance(result, dict) and result.get("error"):
                    status = "failure"
            except asyncio.TimeoutError:
                logger.error(f"❌ Agent {agent_name} 执行超时 ({timeout}s)")
                result = {"error": f"Agent {agent_name} timed out after {timeout}s"}
                status = "failure"
            except Exception as e:
                logger.error(f"❌ Agent {agent_name} 执行失败: {e}")
                result = {"error": str(e)}
                status = "failure"
            finally:
                duration = time.time() - start
                record_agent_execution(agent_name, duration, status)
            return result
