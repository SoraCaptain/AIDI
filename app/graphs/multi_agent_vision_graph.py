# app/graphs/multi_agent_vision_graph.py
# 1. Quality Agent
# 2. OCR Agent
# 3. Detection Agent
# 4. Segmentation Agent
# 5. VLM Agent
import os
import json
import asyncio
from typing import TypedDict, Optional, List, Dict, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from app.agents.local_models import get_text_llm
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.memory.memory_manager import MemoryManager
from app.observability.langfuse_client import (
    get_langfuse_handler,
    build_trace_metadata,
    flush_langfuse,
)
from utils.logger import logger

load_dotenv()


class MultiAgentVisionState(TypedDict, total=False):
    # request
    session_id: str
    question: str
    image_path: Optional[str]

    # memory
    conversation_history: List[Dict[str, Any]]
    last_result: Optional[str]
    memory_context: Dict[str, Any]
    memory_stats: Dict[str, Any]

    # planner
    plan: Dict[str, Any]
    required_agents: List[str]
    planner_reason: str

    # per-agent outputs
    quality_result: Optional[str]
    ocr_result: Optional[str]
    detection_result: Optional[str]
    segmentation_result: Optional[str]
    vlm_result: Optional[str]
    memory_result: Optional[str]

    # aggregate
    aggregated_result: Optional[str]

    # critic / HITL
    critic_decision: Optional[str]
    critic_reason: Optional[str]
    human_decision: Optional[str]
    human_feedback: Optional[str]
    human_edited_answer: Optional[str]

    # control
    retry_count: int
    max_retries: int
    error: Optional[str]

    # final
    final_answer: Optional[str]
    task_id: Optional[str]
    
    
def safe_json_loads(text: str) -> dict:
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


def has_agent(state: MultiAgentVisionState, agent_name: str) -> bool:
    return agent_name in state.get("required_agents", [])


def select_tools(mcp_tools, keywords: List[str]):
    """
    根据工具名关键词筛选工具。
    MCP tool_name_prefix=True 时，工具名通常类似：
    vision_detect_blur
    vision_ocr_image
    vision_grounding_detect
    """
    selected = []

    for tool in mcp_tools:
        name = tool.name.lower()
        if any(k.lower() in name for k in keywords):
            selected.append(tool)

    return selected


def make_load_memory_node(memory_manager: MemoryManager):
    def load_memory_node(state: MultiAgentVisionState) -> dict:
        question = state.get("question", "")
        image_path = state.get("image_path")

        memory_context = memory_manager.build_memory_context(
            question=question,
            image_path=image_path,
        )

        memory_stats = {
            "recent_tasks_count": len(memory_context.get("recent_tasks", [])),
            "same_image_tasks_count": len(memory_context.get("same_image_tasks", [])),
            "keyword_tasks_count": len(memory_context.get("keyword_tasks", [])),
            "similar_tasks_count": len(memory_context.get("similar_tasks", [])),
            "similar_images_count": len(memory_context.get("similar_images", [])),
        }

        return {
            "session_id": memory_manager.session_id,
            "memory_context": memory_context,
            "conversation_history": memory_context.get("conversation_history", []),
            "last_result": memory_context.get("last_result"),
            "memory_stats": memory_stats,
        }

    return load_memory_node


async def planner_node(state: MultiAgentVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    image_path = state.get("image_path")
    question = state.get("question", "")
    memory_context = state.get("memory_context", {})

    system_prompt = """
        你是多 Agent 视觉系统的 Planner Agent。

        你只负责决定需要哪些专职 Agent，不直接分析图片。

        可选 Agent:
        - quality: 图像质量、清晰度、模糊、曝光、尺寸
        - ocr: 图片文字、序列号、标签、表格、文档、屏幕文字
        - detection: 目标检测、bbox、YOLO、GroundingDINO、缺陷定位、scratch/crack/stain/defect
        - segmentation: 分割、mask、轮廓、区域、面积、形状
        - vlm: 高层语义理解、场景解释、综合视觉问答
        - memory: 需要参考历史相似文本任务或相似图片案例

        输出必须是 JSON，不要 Markdown。

        格式:
        {
        "required_agents": ["quality", "ocr", "detection", "segmentation", "vlm", "memory"],
        "plan": {
            "quality": "...",
            "ocr": "...",
            "detection": "...",
            "segmentation": "...",
            "vlm": "...",
            "memory": "..."
        },
        "reason": "..."
        }

        规则:
        1. 如果没有图片，required_agents 只能是 ["memory"] 或 []。
        2. 如果用户问“是否模糊/清晰/质量”，必须包含 quality。
        3. 如果用户问“文字/序列号/标签/读数”，必须包含 ocr。
        4. 如果用户问“位置/bbox/目标/缺陷/scratch/crack/stain”，必须包含 detection。
        5. 如果用户问“区域/轮廓/mask/面积/形状”，必须包含 segmentation。
        6. 如果用户问整体内容、异常解释、综合判断，包含 vlm。
        7. 如果用户提到“历史/以前/类似/上次/相似图片”，包含 memory。
        8. 如果任务复杂，可以包含多个 Agent。
        """

    user_prompt = f"""
        当前图片:
        {image_path}

        用户问题:
        {question}

        最近对话:
        {state.get("conversation_history", [])}

        长期记忆统计:
        {state.get("memory_stats", {})}

        可参考长期记忆:
        similar_tasks:
        {memory_context.get("similar_tasks", [])}

        similar_images:
        {memory_context.get("similar_images", [])}
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
            "run_name": "planner_agent",
            "tags": ["planner", "multi-agent"],
        },
    )

    parsed = safe_json_loads(response.content)

    required_agents = parsed.get("required_agents", [])
    if not isinstance(required_agents, list):
        required_agents = []

    valid = {"quality", "ocr", "detection", "segmentation", "vlm", "memory"}
    required_agents = [a for a in required_agents if a in valid]

    if not image_path:
        required_agents = [a for a in required_agents if a == "memory"]

    return {
        "required_agents": required_agents,
        "plan": parsed.get("plan", {}),
        "planner_reason": parsed.get("reason", ""),
    }
    

def make_quality_agent_node(mcp_tools):
    async def quality_agent_node(state: MultiAgentVisionState) -> dict:
        if not has_agent(state, "quality"):
            return {"quality_result": "SKIPPED: quality agent not required."}

        tools = select_tools(mcp_tools, ["inspect", "blur"])

        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 Quality Agent，只负责图像质量分析。

                必须优先使用工具:
                - inspect_image: 获取图片尺寸和基本信息
                - detect_blur: 判断是否模糊

                不要分析图像内容、缺陷语义或 OCR。
                输出中文，说明调用了哪些工具和结论。
                """,
                        )

        user_message = f"""
            图片路径:
            {state.get("image_path")}

            用户问题:
            {state.get("question")}

            Planner 对 quality 的计划:
            {state.get("plan", {}).get("quality")}
            """

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "run_name": "quality_agent",
                    "tags": ["quality", "mcp-tools"],
                },
            )

            return {"quality_result": result["messages"][-1].content}

        except Exception as e:
            return {
                "quality_result": None,
                "error": f"quality_agent failed: {repr(e)}",
            }

    return quality_agent_node


def make_ocr_agent_node(mcp_tools):
    async def ocr_agent_node(state: MultiAgentVisionState) -> dict:
        if not has_agent(state, "ocr"):
            return {"ocr_result": "SKIPPED: ocr agent not required."}

        tools = select_tools(mcp_tools, ["ocr"])

        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 OCR Agent，只负责图片文字提取和解释。

                必须使用 ocr_image。
                适用于：
                - 标签
                - 序列号
                - 文档
                - 表格
                - 仪表读数
                - UI文字

                不要做目标检测或缺陷判断。
                输出中文，保留识别文本和不确定项。
                """,
        )

        user_message = f"""
            图片路径:
            {state.get("image_path")}

            用户问题:
            {state.get("question")}

            Planner 对 OCR 的计划:
            {state.get("plan", {}).get("ocr")}
            """

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "run_name": "ocr_agent",
                    "tags": ["ocr", "mcp-tools"],
                },
            )

            return {"ocr_result": result["messages"][-1].content}

        except Exception as e:
            return {
                "ocr_result": None,
                "error": f"ocr_agent failed: {repr(e)}",
            }

    return ocr_agent_node


def make_detection_agent_node(mcp_tools):
    async def detection_agent_node(state: MultiAgentVisionState) -> dict:
        if not has_agent(state, "detection"):
            return {"detection_result": "SKIPPED: detection agent not required."}

        tools = select_tools(mcp_tools, ["yolo", "grounding"])

        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 Detection Agent，只负责目标/缺陷检测和定位。

                工具选择:
                - detect_objects_yolo: 通用常见物体检测
                - grounding_detect: 开放词汇检测，适合 scratch、crack、stain、defect、logo、screw 等自定义目标

                如果用户提到 scratch/crack/stain/defect，优先使用 grounding_detect。
                grounding_detect 的 text_prompt 建议用英文短语并用句点分隔，例如:
                "scratch . crack . stain . defect ."

                输出中文，必须包含：
                - 使用的工具
                - 检测到的对象/缺陷
                - bbox 或位置
                - 置信度
                - 不确定性
                """,
        )

        user_message = f"""
            图片路径:
            {state.get("image_path")}

            用户问题:
            {state.get("question")}

            Planner 对 detection 的计划:
            {state.get("plan", {}).get("detection")}

            历史相似图片:
            {state.get("memory_context", {}).get("similar_images", [])}
            """

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "run_name": "detection_agent",
                    "tags": ["detection", "mcp-tools"],
                },
            )

            return {"detection_result": result["messages"][-1].content}

        except Exception as e:
            return {
                "detection_result": None,
                "error": f"detection_agent failed: {repr(e)}",
            }

    return detection_agent_node


def make_segmentation_agent_node(mcp_tools):
    async def segmentation_agent_node(state: MultiAgentVisionState) -> dict:
        if not has_agent(state, "segmentation"):
            return {"segmentation_result": "SKIPPED: segmentation agent not required."}

        tools = select_tools(mcp_tools, ["segment", "sam"])

        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 Segmentation Agent，只负责区域、mask、轮廓、面积、形状分析。

                必须使用 segment_with_sam。
                注意：
                - SAM 自动分割返回的是候选区域，不代表语义类别。
                - 不要把 mask 区域直接断言成缺陷。
                - 如需语义解释，应交给 VLM 或 Report 汇总。

                输出中文，包含：
                - mask 数量
                - 最大区域
                - bbox
                - area
                - 稳定性指标
                """,
        )

        user_message = f"""
            图片路径:
            {state.get("image_path")}

            用户问题:
            {state.get("question")}

            Planner 对 segmentation 的计划:
            {state.get("plan", {}).get("segmentation")}
            """

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "run_name": "segmentation_agent",
                    "tags": ["segmentation", "mcp-tools"],
                },
            )

            return {"segmentation_result": result["messages"][-1].content}

        except Exception as e:
            return {
                "segmentation_result": None,
                "error": f"segmentation_agent failed: {repr(e)}",
            }

    return segmentation_agent_node


def make_vlm_agent_node(mcp_tools):
    async def vlm_agent_node(state: MultiAgentVisionState) -> dict:
        if not has_agent(state, "vlm"):
            return {"vlm_result": "SKIPPED: vlm agent not required."}

        tools = select_tools(mcp_tools, ["vlm"])

        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 VLM Agent，负责高层图像语义理解。

                必须使用 ask_vlm。
                你可以参考其他 Agent 的结果，但不能编造图片内容。
                如果其他工具结果与 VLM 观察冲突，请说明冲突。
                输出中文。
                """,
        )

        user_message = f"""
            图片路径:
            {state.get("image_path")}

            用户问题:
            {state.get("question")}

            Quality Result:
            {state.get("quality_result")}

            OCR Result:
            {state.get("ocr_result")}

            Detection Result:
            {state.get("detection_result")}

            Segmentation Result:
            {state.get("segmentation_result")}

            Planner 对 VLM 的计划:
            {state.get("plan", {}).get("vlm")}
            """

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "run_name": "vlm_agent",
                    "tags": ["vlm", "mcp-tools"],
                },
            )

            return {"vlm_result": result["messages"][-1].content}

        except Exception as e:
            return {
                "vlm_result": None,
                "error": f"vlm_agent failed: {repr(e)}",
            }

    return vlm_agent_node


async def memory_agent_node(state: MultiAgentVisionState) -> dict:
    """Memory Agent 不调用 MCP，只负责整理历史记忆。"""
    if not has_agent(state, "memory"):
        return {"memory_result": "SKIPPED: memory agent not required."}

    llm = get_text_llm(temperature=0)

    memory_context = state.get("memory_context", {})

    system_prompt = """
        你是 Memory Agent，只负责总结历史记忆。

        你需要区分：
        - same_image_tasks: 同一图片历史
        - similar_tasks: 文本相似历史任务
        - similar_images: 图像相似历史案例

        不要把历史案例当作当前图片的新观察。
        输出中文，说明历史参考的可靠性和局限。
        """

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        当前图片:
        {state.get("image_path")}

        same_image_tasks:
        {memory_context.get("same_image_tasks", [])}

        similar_tasks:
        {memory_context.get("similar_tasks", [])}

        similar_images:
        {memory_context.get("similar_images", [])}
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
            "run_name": "memory_agent",
            "tags": ["memory-agent"],
        },
    )

    return {"memory_result": response.content}


async def aggregator_node(state: MultiAgentVisionState) -> dict:
    """把所有子 Agent 输出汇总成一个结构化上下文，交给 Critic 和 Report。
    """
    llm = get_text_llm(temperature=0)

    system_prompt = """
        你是 Aggregator。

        你负责把多个专职 Agent 的结果整合成一个中间分析结果。
        不要引入新观察，只能汇总已有 Agent 输出。
        如果不同 Agent 之间有冲突，请明确列出。
        """

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        Planner:
        required_agents = {state.get("required_agents")}
        reason = {state.get("planner_reason")}

        Quality Result:
        {state.get("quality_result")}

        OCR Result:
        {state.get("ocr_result")}

        Detection Result:
        {state.get("detection_result")}

        Segmentation Result:
        {state.get("segmentation_result")}

        VLM Result:
        {state.get("vlm_result")}

        Memory Result:
        {state.get("memory_result")}

        Error:
        {state.get("error")}
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
            "run_name": "aggregator",
            "tags": ["aggregator"],
        },
    )

    return {"aggregated_result": response.content}


async def critic_node(state: MultiAgentVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    system_prompt = """
        你是 Critic Agent。

        你需要审核多 Agent 汇总结果是否足够可靠。

        可选 decision:
        - pass: 可以生成报告
        - retry: 缺少关键信息，但可以自动重试
        - human_review: 结果不确定、涉及关键缺陷判断、工具冲突、或需要人工复核
        - fail: 无法完成任务

        输出 JSON:
        {
        "decision": "pass|retry|human_review|fail",
        "reason": "..."
        }

        触发 human_review:
        1. 检测到疑似缺陷但置信度低
        2. 工具结果冲突
        3. VLM 与 detection 结论不一致
        4. 用户显式要求人工确认
        5. 自动重试达到上限
        """

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        Aggregated Result:
        {state.get("aggregated_result")}

        Quality:
        {state.get("quality_result")}

        OCR:
        {state.get("ocr_result")}

        Detection:
        {state.get("detection_result")}

        Segmentation:
        {state.get("segmentation_result")}

        VLM:
        {state.get("vlm_result")}

        Memory:
        {state.get("memory_result")}

        retry_count:
        {retry_count}

        max_retries:
        {max_retries}
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
            "run_name": "critic_agent",
            "tags": ["critic", "quality-gate"],
        },
    )

    parsed = safe_json_loads(response.content)

    decision = parsed.get("decision", "pass")
    reason = parsed.get("reason", "")

    if decision == "retry" and retry_count >= max_retries:
        decision = "human_review"
        reason = reason + "；已达到最大自动重试次数，转人工复核。"

    if decision not in ["pass", "retry", "human_review", "fail"]:
        decision = "pass"

    return {
        "critic_decision": decision,
        "critic_reason": reason,
    }
    
    
def human_review_node(state: MultiAgentVisionState) -> dict:
    review_payload = {
        "type": "multi_agent_vision_review_required",
        "message": "需要人工复核多 Agent 视觉分析结果。",
        "question": state.get("question"),
        "image_path": state.get("image_path"),
        "aggregated_result": state.get("aggregated_result"),
        "quality_result": state.get("quality_result"),
        "ocr_result": state.get("ocr_result"),
        "detection_result": state.get("detection_result"),
        "segmentation_result": state.get("segmentation_result"),
        "vlm_result": state.get("vlm_result"),
        "memory_result": state.get("memory_result"),
        "critic_decision": state.get("critic_decision"),
        "critic_reason": state.get("critic_reason"),
        "retry_count": state.get("retry_count", 0),
        "allowed_actions": [
            {"action": "accept", "description": "接受结果并生成报告"},
            {"action": "edit", "description": "人工修改最终分析", "fields": ["edited_answer"]},
            {"action": "retry", "description": "带反馈重试", "fields": ["feedback"]},
            {"action": "reject", "description": "拒绝当前分析", "fields": ["feedback"]},
        ],
    }

    human_response = interrupt(review_payload)

    action = human_response.get("action")
    feedback = human_response.get("feedback", "")
    edited_answer = human_response.get("edited_answer")

    if action == "accept":
        return {
            "human_decision": "accept",
            "human_feedback": feedback,
        }

    if action == "edit":
        return {
            "human_decision": "edit",
            "human_feedback": feedback,
            "human_edited_answer": edited_answer,
            "aggregated_result": edited_answer,
        }

    if action == "retry":
        return {
            "human_decision": "retry",
            "human_feedback": feedback,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    if action == "reject":
        return {
            "human_decision": "reject",
            "human_feedback": feedback,
            "critic_decision": "fail",
        }

    return {
        "human_decision": "reject",
        "human_feedback": f"未知人工动作: {action}",
        "critic_decision": "fail",
    }
    
    
async def report_node(state: MultiAgentVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    system_prompt = """
        你是 Report Agent。

        请根据多 Agent 结果生成最终视觉分析报告。

        要求:
        - 中文
        - 结构清晰
        - 区分当前工具观察、历史相似案例、人工复核
        - 不要编造未被工具或人工确认的信息
        - 如果有人工 edit，以人工 edit 为准
        - 如果分析失败，说明失败原因和下一步建议
        """

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        图片:
        {state.get("image_path")}

        Planner:
        required_agents = {state.get("required_agents")}
        reason = {state.get("planner_reason")}

        Quality:
        {state.get("quality_result")}

        OCR:
        {state.get("ocr_result")}

        Detection:
        {state.get("detection_result")}

        Segmentation:
        {state.get("segmentation_result")}

        VLM:
        {state.get("vlm_result")}

        Memory:
        {state.get("memory_result")}

        Aggregated:
        {state.get("aggregated_result")}

        Critic:
        decision = {state.get("critic_decision")}
        reason = {state.get("critic_reason")}

        Human:
        decision = {state.get("human_decision")}
        feedback = {state.get("human_feedback")}
        edited_answer = {state.get("human_edited_answer")}
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
            "run_name": "report_agent",
            "tags": ["report"],
        },
    )

    return {"final_answer": response.content}


def make_save_memory_node(memory_manager: MemoryManager):
    def save_memory_node(state: MultiAgentVisionState) -> dict:
        # 兼容之前 PersistentMemory 字段：
        # 把多 Agent 汇总结果放到 vision_answer。
        save_state = dict(state)
        save_state["vision_answer"] = state.get("aggregated_result")
        save_state["task_type"] = ",".join(state.get("required_agents", []))
        save_state["planner_reason"] = state.get("planner_reason")

        task_id = memory_manager.save_graph_result(save_state)

        return {"task_id": task_id}

    return save_memory_node


def route_after_critic(state: MultiAgentVisionState) -> str:
    decision = state.get("critic_decision")

    if decision == "retry":
        return "increment_retry"

    if decision == "human_review":
        return "human_review"

    return "report"


def increment_retry_node(state: MultiAgentVisionState) -> dict:
    return {
        "retry_count": state.get("retry_count", 0) + 1
    }


def route_after_human_review(state: MultiAgentVisionState) -> str:
    if state.get("human_decision") == "retry":
        return "quality_agent"

    return "report"


def build_multi_agent_vision_graph(mcp_tools, memory_manager: MemoryManager):
    graph = StateGraph(MultiAgentVisionState)

    graph.add_node("load_memory", make_load_memory_node(memory_manager))
    graph.add_node("planner", planner_node)

    graph.add_node("quality_agent", make_quality_agent_node(mcp_tools))
    graph.add_node("ocr_agent", make_ocr_agent_node(mcp_tools))
    graph.add_node("detection_agent", make_detection_agent_node(mcp_tools))
    graph.add_node("segmentation_agent", make_segmentation_agent_node(mcp_tools))
    graph.add_node("vlm_agent", make_vlm_agent_node(mcp_tools))
    graph.add_node("memory_agent", memory_agent_node)

    graph.add_node("aggregator", aggregator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("report", report_node)
    graph.add_node("save_memory", make_save_memory_node(memory_manager))

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "planner")

    # 顺序执行专职 Agent；每个 Agent 内部自己判断是否 SKIP
    graph.add_edge("planner", "quality_agent")
    graph.add_edge("quality_agent", "ocr_agent")
    graph.add_edge("ocr_agent", "detection_agent")
    graph.add_edge("detection_agent", "segmentation_agent")
    graph.add_edge("segmentation_agent", "vlm_agent")
    graph.add_edge("vlm_agent", "memory_agent")

    graph.add_edge("memory_agent", "aggregator")
    graph.add_edge("aggregator", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "increment_retry": "increment_retry",
            "human_review": "human_review",
            "report": "report",
        },
    )

    # retry 后从 quality_agent 重新跑一遍子 Agent 链
    graph.add_edge("increment_retry", "quality_agent")

    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "quality_agent": "quality_agent",
            "report": "report",
        },
    )

    graph.add_edge("report", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)


async def run_one_turn(app, memory_manager: MemoryManager, thread_id: str):
    image_path = input("\nImage path, empty if same as before: ").strip()

    if image_path.lower() == "exit":
        return False

    while (image_path and not os.path.isfile(image_path)):
        image_path = input("\nValid image path, enter again: ").strip()
    if image_path and os.path.isfile(image_path):
        memory_manager.set_current_image(image_path)

    question = input("Question: ").strip()

    if question.lower() == "exit":
        return False

    current_image_path = memory_manager.get_current_image()

    memory_manager.add_user_message(question)

    initial_state = {
        "session_id": memory_manager.session_id,
        "question": question,
        "image_path": current_image_path,
        "retry_count": 0,
        "max_retries": 2,
    }

    langfuse_handler = get_langfuse_handler()

    trace_metadata = build_trace_metadata(
        session_id=memory_manager.session_id,
        thread_id=thread_id,
        image_path=current_image_path,
        question=question,
        extra={
            "graph": "multi_agent_vision_graph",
        },
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "callbacks": [langfuse_handler],
        "metadata": trace_metadata,
        "tags": [
            "vision-agent",
            "multi-agent",
            "langgraph",
            "mcp",
            "memory",
            "hitl",
        ],
    }

    result = await app.ainvoke(initial_state, config=config)

    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value

        logger.info("\n" + "=" * 80)
        logger.info("需要人工复核：")
        logger.info(json.dumps(interrupt_value, ensure_ascii=False, indent=2))
        logger.info("=" * 80)

        logger.info("\n请选择人工动作：")
        logger.info("1. accept")
        logger.info("2. edit")
        logger.info("3. retry")
        logger.info("4. reject")

        action = input("Action: ").strip()

        if action == "1":
            resume_value = {
                "action": "accept",
                "feedback": "人工接受当前结果。",
            }
        elif action == "2":
            edited_answer = input("Edited answer: ").strip()
            resume_value = {
                "action": "edit",
                "feedback": "人工修改了多 Agent 分析结果。",
                "edited_answer": edited_answer,
            }
        elif action == "3":
            feedback = input("Retry feedback: ").strip()
            resume_value = {
                "action": "retry",
                "feedback": feedback,
            }
        elif action == "4":
            feedback = input("Reject reason: ").strip()
            resume_value = {
                "action": "reject",
                "feedback": feedback,
            }
        else:
            resume_value = {
                "action": "reject",
                "feedback": f"无效动作: {action}",
            }

        result = await app.ainvoke(
            Command(resume=resume_value),
            config=config,
        )

    final_answer = result.get("final_answer", "")
    task_id = result.get("task_id")

    memory_manager.add_assistant_message(final_answer)

    logger.info("\n" + "=" * 80)
    logger.info(final_answer)
    logger.info(f"\n保存到长期记忆 task_id: {task_id}")
    logger.info("=" * 80)

    logger.info("\nTrace summary:")
    logger.info(
        json.dumps(
            {
                "required_agents": result.get("required_agents"),
                "critic_decision": result.get("critic_decision"),
                "human_decision": result.get("human_decision"),
                "retry_count": result.get("retry_count"),
                "task_id": task_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return True


async def main():
    try:
        mcp_client, mcp_tools = await load_vision_mcp_tools()

        logger.info("Loaded MCP tools:")
        for tool in mcp_tools:
            logger.info(f"- {tool.name}: {tool.description[:100]}")

        session_id = "multi-agent-vision-session-001"

        memory_manager = MemoryManager(
            session_id=session_id,
            db_path="data/memory/vision_memory.sqlite3",
            max_turns=8,
            enable_vector_memory=True,
            enable_image_vector_memory=True,
        )

        app = build_multi_agent_vision_graph(
            mcp_tools=mcp_tools,
            memory_manager=memory_manager,
        )

        logger.info("\nMulti-Agent Vision Graph started.")
        logger.info("输入 exit 退出。")

        thread_id = session_id

        while True:
            should_continue = await run_one_turn(
                app=app,
                memory_manager=memory_manager,
                thread_id=thread_id,
            )

            if not should_continue:
                break

    finally:
        flush_langfuse()


if __name__ == "__main__":
    asyncio.run(main())
    
    
# /home/ziyi/gitlocal/AIDI/test_imgs/train01.png
# /home/ziyi/gitlocal/AIDI/test_imgs/WDED1900240A_04-Cam2-85-1.bmp
# 请检测图中是否有长条状污渍，并解释是否可能是缺陷。
# /home/ziyi/gitlocal/AIDI/test_imgs/WDLD13078D2A_03-Cam1-158-2.bmp
# 请定位疑似缺陷区域，并给出区域轮廓和面积。
# /home/ziyi/gitlocal/AIDI/test_imgs/WDLD14249F1A_04-Cam2-1150-3.bmp
# 这张图是否模糊？清晰度怎么样？
# /home/ziyi/gitlocal/AIDI/test_imgs/WDLD14439B1A_16-Cam1-1226-3.bmp
# 这张图有没有和以前类似的缺陷
# /home/ziyi/gitlocal/AIDI/test_imgs/WDLD13055F1A_15-Cam1-765-4.bmp
# 这张图和历史相似图片相比，有没有类似问题？
# /home/ziyi/gitlocal/AIDI/test_imgs/WDPD2121940A_13-Cam1-13-4.bmp
# 请读取这张图里的文字
