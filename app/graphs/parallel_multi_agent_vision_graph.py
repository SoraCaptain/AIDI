# app/graphs/parallel_multi_agent_vision_graph.py
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

load_dotenv()


class ParallelVisionState(TypedDict, total=False):
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

    # agent outputs
    quality_result: Optional[str]
    ocr_result: Optional[str]
    detection_result: Optional[str]
    segmentation_result: Optional[str]
    vlm_result: Optional[str]
    memory_result: Optional[str]

    # agent errors
    quality_error: Optional[str]
    ocr_error: Optional[str]
    detection_error: Optional[str]
    segmentation_error: Optional[str]
    vlm_error: Optional[str]
    memory_error: Optional[str]

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


def has_agent(state: ParallelVisionState, agent_name: str) -> bool:
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
    def load_memory_node(state: ParallelVisionState) -> dict:
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


async def planner_node(state: ParallelVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    image_path = state.get("image_path")
    question = state.get("question", "")
    memory_context = state.get("memory_context", {})

    system_prompt = """
        你是并行多 Agent 视觉系统的 Planner Agent。

        你只负责决定需要启动哪些专职 Agent，不直接分析图片。

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
        1. 如果没有图片，required_agents 只能包含 memory 或为空。
        2. 如果用户问模糊/清晰/质量，必须包含 quality。
        3. 如果用户问文字/序列号/标签/读数，必须包含 ocr。
        4. 如果用户问位置/bbox/目标/缺陷/scratch/crack/stain，必须包含 detection。
        5. 如果用户问区域/轮廓/mask/面积/形状，必须包含 segmentation。
        6. 如果用户问整体内容、异常解释、综合判断，包含 vlm。
        7. 如果用户提到历史/以前/类似/上次/相似图片，包含 memory。
        8. 复杂任务可以包含多个 Agent。
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
            "run_name": "planner_agent_parallel",
            "tags": ["planner", "parallel"],
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
 

def dispatch_agents(state: ParallelVisionState) -> List:
    """
    根据 Planner 输出，返回要并行启动的节点列表。

    注意：
    - 返回多个节点名即可 fan-out。
    - 如果没有任何 Agent 需要执行，直接进入 aggregator。
    """

    required = set(state.get("required_agents", []))

    destinations = []

    if "quality" in required:
        destinations.append("quality_agent")

    if "ocr" in required:
        destinations.append("ocr_agent")

    if "detection" in required:
        destinations.append("detection_agent")

    if "segmentation" in required:
        destinations.append("segmentation_agent")

    if "vlm" in required:
        destinations.append("vlm_agent")

    if "memory" in required:
        destinations.append("memory_agent")

    if not destinations:
        destinations.append("aggregator")

    return destinations


def make_quality_agent_node(mcp_tools):
    async def quality_agent_node(state: ParallelVisionState) -> dict:
        tools = select_tools(mcp_tools, ["inspect", "blur"])
        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 Quality Agent，只负责图像质量分析。

                必须优先使用：
                - inspect_image
                - detect_blur

                不要分析 OCR、目标检测或语义缺陷。
                输出中文，说明工具、指标和结论。
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
                    "run_name": "quality_agent_parallel",
                    "tags": ["quality", "parallel", "mcp-tools"],
                },
            )

            return {"quality_result": result["messages"][-1].content}

        except Exception as e:
            return {"quality_error": repr(e)}

    return quality_agent_node


def make_ocr_agent_node(mcp_tools):
    async def ocr_agent_node(state: ParallelVisionState) -> dict:
        tools = select_tools(mcp_tools, ["ocr"])
        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 OCR Agent，只负责图片文字提取。

                必须使用 ocr_image。
                不要做缺陷判断或目标检测。
                输出中文，保留识别文本、不确定项和可能的误识别。
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
                    "run_name": "ocr_agent_parallel",
                    "tags": ["ocr", "parallel", "mcp-tools"],
                },
            )

            return {"ocr_result": result["messages"][-1].content}

        except Exception as e:
            return {"ocr_error": repr(e)}

    return ocr_agent_node


def make_detection_agent_node(mcp_tools):
    async def detection_agent_node(state: ParallelVisionState) -> dict:
        tools = select_tools(mcp_tools, ["yolo", "grounding"])
        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 Detection Agent，只负责目标或缺陷定位。

                工具选择：
                - detect_objects_yolo: 常见物体检测
                - grounding_detect: 开放词汇检测，适合 scratch、crack、stain、defect 等

                如果用户提到 scratch/crack/stain/defect，优先使用 grounding_detect。
                grounding_detect 的 text_prompt 使用英文短语并用句点分隔，例如:
                "scratch . crack . stain . defect ."

                输出中文，包含工具、bbox、置信度、不确定性。
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
                    "run_name": "detection_agent_parallel",
                    "tags": ["detection", "parallel", "mcp-tools"],
                },
            )

            return {"detection_result": result["messages"][-1].content}

        except Exception as e:
            return {"detection_error": repr(e)}

    return detection_agent_node


def make_segmentation_agent_node(mcp_tools):
    async def segmentation_agent_node(state: ParallelVisionState) -> dict:
        tools = select_tools(mcp_tools, ["segment", "sam"])
        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 Segmentation Agent，只负责区域、mask、轮廓、面积、形状分析。

                必须使用 segment_with_sam。
                注意：
                - SAM 输出是候选区域，不代表语义类别。
                - 不要直接断言某个 mask 是缺陷。
                - 输出 bbox、area、stability_score 等。
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
                    "run_name": "segmentation_agent_parallel",
                    "tags": ["segmentation", "parallel", "mcp-tools"],
                },
            )

            return {"segmentation_result": result["messages"][-1].content}

        except Exception as e:
            return {"segmentation_error": repr(e)}

    return segmentation_agent_node


def make_vlm_agent_node(mcp_tools):
    async def vlm_agent_node(state: ParallelVisionState) -> dict:
        tools = select_tools(mcp_tools, ["vlm"])
        llm = get_text_llm(temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
                你是 VLM Agent，负责高层图像语义理解。

                必须使用 ask_vlm。
                你不依赖其他并行 Agent 的结果，因为你和它们并行执行。
                不要编造图片内容。
                输出中文。
                """,
        )

        user_message = f"""
            图片路径:
            {state.get("image_path")}

            用户问题:
            {state.get("question")}

            Planner 对 VLM 的计划:
            {state.get("plan", {}).get("vlm")}
            """

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "run_name": "vlm_agent_parallel",
                    "tags": ["vlm", "parallel", "mcp-tools"],
                },
            )

            return {"vlm_result": result["messages"][-1].content}

        except Exception as e:
            return {"vlm_error": repr(e)}

    return vlm_agent_node


async def memory_agent_node(state: ParallelVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    memory_context = state.get("memory_context", {})

    system_prompt = """
        你是 Memory Agent，只负责总结历史记忆。

        区分：
        - same_image_tasks: 同一图片历史
        - similar_tasks: 文本相似任务
        - similar_images: 图像相似案例

        不要把历史案例当成当前图片的新观察。
        输出中文，说明历史参考价值和局限。
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

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            config={
                "run_name": "memory_agent_parallel",
                "tags": ["memory-agent", "parallel"],
            },
        )

        return {"memory_result": response.content}

    except Exception as e:
        return {"memory_error": repr(e)}


async def aggregator_node(state: ParallelVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    system_prompt = """
        你是 Aggregator，负责把并行 Agent 的输出汇总成中间分析结果。

        要求:
        1. 不引入新观察。
        2. 只汇总已有 Agent 输出。
        3. 明确列出哪些 Agent 执行了，哪些没有执行。
        4. 如果结果冲突，列出冲突。
        5. 如果某个 Agent 报错，说明错误但不要让整个任务失败。
        6. 区分当前工具观察和历史记忆。
        """

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        图片:
        {state.get("image_path")}

        Planner:
        required_agents = {state.get("required_agents")}
        reason = {state.get("planner_reason")}
        plan = {state.get("plan")}

        Quality Result:
        {state.get("quality_result")}
        Quality Error:
        {state.get("quality_error")}

        OCR Result:
        {state.get("ocr_result")}
        OCR Error:
        {state.get("ocr_error")}

        Detection Result:
        {state.get("detection_result")}
        Detection Error:
        {state.get("detection_error")}

        Segmentation Result:
        {state.get("segmentation_result")}
        Segmentation Error:
        {state.get("segmentation_error")}

        VLM Result:
        {state.get("vlm_result")}
        VLM Error:
        {state.get("vlm_error")}

        Memory Result:
        {state.get("memory_result")}
        Memory Error:
        {state.get("memory_error")}
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
            "run_name": "parallel_aggregator",
            "tags": ["aggregator", "map-reduce"],
        },
    )

    return {"aggregated_result": response.content}


async def critic_node(state: ParallelVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    system_prompt = """
        你是 Critic Agent。

        你需要审核并行多 Agent 汇总结果是否足够可靠。

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
        """

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        Aggregated Result:
        {state.get("aggregated_result")}

        Agent Errors:
        quality_error={state.get("quality_error")}
        ocr_error={state.get("ocr_error")}
        detection_error={state.get("detection_error")}
        segmentation_error={state.get("segmentation_error")}
        vlm_error={state.get("vlm_error")}
        memory_error={state.get("memory_error")}

        retry_count:
        {retry_count}

        max_retries:
        {max_retries}

        判断规则:
        1. 如果关键 Agent 失败且无法回答用户问题，可以 retry。
        2. 如果 detection/VLM/segmentation 之间冲突，human_review。
        3. 如果涉及疑似缺陷但置信度不足，human_review。
        4. 如果已达到最大重试次数仍不可靠，human_review。
        5. 如果信息充分，pass。
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
            "run_name": "critic_agent_parallel",
            "tags": ["critic", "parallel"],
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
  
    
def human_review_node(state: ParallelVisionState) -> dict:
    review_payload = {
        "type": "parallel_multi_agent_review_required",
        "message": "需要人工复核并行多 Agent 视觉分析结果。",
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
    
   
async def report_node(state: ParallelVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    system_prompt = """
        你是 Report Agent。

        请根据并行多 Agent 结果生成最终视觉分析报告。

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
            "run_name": "report_agent_parallel",
            "tags": ["report", "parallel"],
        },
    )

    return {"final_answer": response.content}  
   

def prepare_retry_node(state: ParallelVisionState) -> dict:
    return {
        "quality_result": None,
        "ocr_result": None,
        "detection_result": None,
        "segmentation_result": None,
        "vlm_result": None,
        "memory_result": None,
        "quality_error": None,
        "ocr_error": None,
        "detection_error": None,
        "segmentation_error": None,
        "vlm_error": None,
        "memory_error": None,
        "aggregated_result": None,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def make_save_memory_node(memory_manager: MemoryManager):
    def save_memory_node(state: ParallelVisionState) -> dict:
        save_state = dict(state)

        save_state["vision_answer"] = state.get("aggregated_result")
        save_state["task_type"] = ",".join(state.get("required_agents", []))
        save_state["planner_reason"] = state.get("planner_reason")

        task_id = memory_manager.save_graph_result(save_state)

        return {"task_id": task_id}

    return save_memory_node


def route_after_critic(state: ParallelVisionState) -> str:
    decision = state.get("critic_decision")

    if decision == "retry":
        return "prepare_retry"

    if decision == "human_review":
        return "human_review"

    return "report"


def route_after_human_review(state: ParallelVisionState) -> str:
    if state.get("human_decision") == "retry":
        return "prepare_retry"

    return "report"


def increment_retry_node(state: ParallelVisionState) -> dict:
    return {
        "retry_count": state.get("retry_count", 0) + 1
    }


def route_after_human_review(state: ParallelVisionState) -> str:
    if state.get("human_decision") == "retry":
        return "quality_agent"

    return "report"


def build_parallel_multi_agent_vision_graph(
    mcp_tools,
    memory_manager: MemoryManager,
    checkpointer=None,
):
    graph = StateGraph(ParallelVisionState)

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
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("report", report_node)
    graph.add_node("save_memory", make_save_memory_node(memory_manager))

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "planner")

    graph.add_conditional_edges(
        "planner",
        dispatch_agents,
        {
            "quality_agent": "quality_agent",
            "ocr_agent": "ocr_agent",
            "detection_agent": "detection_agent",
            "segmentation_agent": "segmentation_agent",
            "vlm_agent": "vlm_agent",
            "memory_agent": "memory_agent",
            "aggregator": "aggregator",
        },
    )

    # 所有并行 worker 汇入 aggregator
    graph.add_edge("quality_agent", "aggregator")
    graph.add_edge("ocr_agent", "aggregator")
    graph.add_edge("detection_agent", "aggregator")
    graph.add_edge("segmentation_agent", "aggregator")
    graph.add_edge("vlm_agent", "aggregator")
    graph.add_edge("memory_agent", "aggregator")

    graph.add_edge("aggregator", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "prepare_retry": "prepare_retry",
            "human_review": "human_review",
            "report": "report",
        },
    )

    graph.add_conditional_edges(
        "prepare_retry",
        dispatch_agents,
        {
            "quality_agent": "quality_agent",
            "ocr_agent": "ocr_agent",
            "detection_agent": "detection_agent",
            "segmentation_agent": "segmentation_agent",
            "vlm_agent": "vlm_agent",
            "memory_agent": "memory_agent",
            "aggregator": "aggregator",
        },
    )

    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "prepare_retry": "prepare_retry",
            "report": "report",
        },
    )

    graph.add_edge("report", "save_memory")
    graph.add_edge("save_memory", END)

    
    if checkpointer is None:
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
            "graph": "parallel_multi_agent_vision_graph",
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

        print("\n" + "=" * 80)
        print("需要人工复核：")
        print(json.dumps(interrupt_value, ensure_ascii=False, indent=2))
        print("=" * 80)

        print("\n请选择人工动作：")
        print("1. accept")
        print("2. edit")
        print("3. retry")
        print("4. reject")

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

    print("\n" + "=" * 80)
    print(final_answer)
    print("\n保存到长期记忆 task_id:", task_id)
    print("=" * 80)

    print("\nTrace summary:")
    print(
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

        print("Loaded MCP tools:")
        for tool in mcp_tools:
            print(f"- {tool.name}: {tool.description[:100]}")

        session_id = "multi-agent-vision-session-001"

        memory_manager = MemoryManager(
            session_id=session_id,
            db_path="data/memory/vision_memory.sqlite3",
            max_turns=8,
            enable_vector_memory=True,
            enable_image_vector_memory=True,
        )

        app = build_parallel_multi_agent_vision_graph(
            mcp_tools=mcp_tools,
            memory_manager=memory_manager,
        )

        print("\nMulti-Agent Vision Graph started.")
        print("输入 exit 退出。")

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
