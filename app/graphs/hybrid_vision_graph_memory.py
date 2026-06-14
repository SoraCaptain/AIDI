# app/graphs/hybrid_vision_graph_hitl.py
import os
import json
import asyncio
from typing import TypedDict, Optional, List, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from app.agents.local_models import get_text_llm
from app.agents.single_agent import (
    QwenToolCallParsingMiddleware,
    ToolUsageReminderMiddleware,
)
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.memory.session_memory import SessionMemory
from app.memory.memory_manager import MemoryManager
from app.observability.langfuse_client import (
    get_langfuse_handler,
    build_trace_metadata,
    flush_langfuse,
)


load_dotenv()


class HybridVisionState(TypedDict, total=False):
    question: str
    image_path: Optional[str]

    session_id: str
    task_id: Optional[str]

    conversation_history: List[Dict]
    last_result: Optional[str]

    memory_context: Optional[Dict]

    plan: Optional[str]
    task_type: Optional[str]
    planner_reason: Optional[str]

    vision_answer: Optional[str]

    critic_decision: Optional[str]
    critic_reason: Optional[str]

    human_decision: Optional[str]
    human_feedback: Optional[str]
    human_edited_answer: Optional[str]

    retry_count: int
    max_retries: int
    error: Optional[str]

    final_answer: Optional[str]

    memory_stats: Optional[Dict]


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


def make_load_memory_node(memory_manager):
    def load_memory_node(state: HybridVisionState) -> dict:
        question = state.get("question", "")
        image_path = state.get("image_path")

        memory_context = memory_manager.build_memory_context(
            question=question,
            image_path=image_path,
        )

        return {
            "memory_context": memory_context,
            "conversation_history": memory_context.get("conversation_history", []),
            "last_result": memory_context.get("last_result"),
            "session_id": memory_manager.session_id,
            "memory_stats": {
                "recent_tasks_count": len(memory_context.get("recent_tasks", [])),
                "same_image_tasks_count": len(memory_context.get("same_image_tasks", [])),
                "keyword_tasks_count": len(memory_context.get("keyword_tasks", [])),
                "similar_tasks_count": len(memory_context.get("similar_tasks", [])),
                "similar_images_count": len(memory_context.get("similar_images", [])),
            },
        }

    return load_memory_node


def _compress_memory_for_planner(
    memory_context: dict,
    conversation_history: list,
    last_result: str | None,
    max_items: int = 5,
    max_len: int = 120,
) -> str:
    """Compress memory context into a compact prompt snippet for the planner."""

    def _trunc(v, max_len=max_len) -> str:
        s = str(v) if v else ""
        return s[:max_len] + ("..." if len(s) > max_len else "")

    def _compact_list(items: list, *keys: str, limit: int = max_items) -> str:
        if not items:
            return "  (无)"
        lines = []
        for i, item in enumerate(items[:limit]):
            parts = " | ".join(f"{k}={_trunc(item.get(k))}" for k in keys)
            lines.append(f"  [{i+1}] {parts}")
        if len(items) > limit:
            lines.append(f"  ...还有 {len(items) - limit} 条已省略")
        return "\n".join(lines)

    parts: list[str] = []

    # 上一轮结果 (truncate)
    if last_result:
        parts.append(f"上一轮结果: {_trunc(last_result, 200)}")

    # 最近对话 (last few turns)
    conv_last = conversation_history[-6:] if conversation_history else []
    if conv_last:
        conv_lines = []
        for m in conv_last:
            role = m.get("role", "?")
            content = _trunc(m.get("content", ""), 80)
            conv_lines.append(f"  [{role}] {content}")
        parts.append("最近对话:\n" + "\n".join(conv_lines))

    # 长期记忆各维度
    sections = [
        ("recent_tasks", "question", "task_type"),
        ("same_image_tasks", "question", "task_type"),
        ("keyword_tasks", "question", "task_type"),
        ("similar_tasks", "question", "task_type"),
        ("similar_images", "question", "image_path", "task_type"),
    ]
    for label, *keys in sections:
        items = memory_context.get(label, []) if memory_context else []
        snip = _compact_list(items, *keys)
        parts.append(f"{label}:\n{snip}")

    return "\n\n".join(parts)


async def planner_node(state: HybridVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    image_path = state.get("image_path")
    question = state.get("question", "")

    system_prompt = """
        你是视觉任务规划 Agent。

        你只负责判断任务类型，不直接分析图片。
        
        如果长期记忆中有同一图片或类似问题的历史结果，可以在 plan 和 reason 中说明需要参考历史结果，但不要直接替代新的视觉分析。
        如果 similar_tasks 中存在高相似历史案例，可以在 plan 中说明要参考它们。但长期记忆只能作为参考，不能替代当前工具分析。
        如果 similar_images 中存在高相似历史图片，可以在 plan 中说明要参考它们。但相似图片只能提供历史参考，不能替代当前图片分析。

        可选 task_type:
        - no_image: 用户没有提供图片，无法做视觉分析
        - quality_check: 用户主要关心清晰度、模糊、曝光、图像质量
        - image_understanding: 用户主要关心图像内容、物体、场景、缺陷、异常
        - both: 同时需要图像质量检测和图像内容理解
        - report_only: 用户只是要求基于已有结果总结或解释

        请只输出 JSON，不要输出 Markdown。

        JSON 格式:
        {
        "task_type": "...",
        "plan": "...",
        "reason": "..."
        }
        """

    memory_snippet = _compress_memory_for_planner(
        memory_context=state.get("memory_context", {}),
        conversation_history=state.get("conversation_history", []),
        last_result=state.get("last_result"),
        max_items=1
    )

    user_prompt = f"""图片: {image_path or '(无)'}
        问题: {question}

        {memory_snippet}"""
    print('debug planner user_prompt', user_prompt)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
                "run_name": "planner_llm",
                "tags": ["planner", "routing"],
        },
    )

    parsed = safe_json_loads(response.content)
    task_type = parsed.get("task_type")

    if not image_path and task_type != "report_only":
        task_type = "no_image"

    valid_task_types = [
        "no_image",
        "quality_check",
        "image_understanding",
        "both",
        "report_only",
    ]

    if task_type not in valid_task_types:
        task_type = "both" if image_path else "no_image"

    return {
        "task_type": task_type,
        "plan": parsed.get("plan", ""),
        "planner_reason": parsed.get("reason", ""),
    }


def route_after_planner(state: HybridVisionState) -> str:
    task_type = state.get("task_type")

    if task_type in ["no_image", "report_only"]:
        return "report"

    return "vision_agent"


def make_vision_agent_node(mcp_tools):
    async def vision_agent_node(state: HybridVisionState) -> dict:
        llm = get_text_llm(temperature=0)

        tool_names = [tool.name for tool in mcp_tools]

        agent = create_agent(
            model=llm,
            tools=mcp_tools,
            # middleware=[
            #     QwenToolCallParsingMiddleware(),
            #     ToolUsageReminderMiddleware(),
            # ],
            system_prompt=f"""
                你是一个视觉分析 Agent。

                你通过 MCP tools 调用内部视觉能力。

                当前可用工具:
                {tool_names}

                工具选择规则:

                1. 基础图片信息
                - 如果需要知道图片尺寸、格式、路径是否可访问，调用 inspect_image。

                2. 图像质量
                - 如果用户问清晰度、模糊、焦点、拍摄质量，调用 detect_blur。

                3. OCR / 文本
                - 如果用户问图片中的文字、标签、序列号、仪表读数、文档内容、表格文字，调用 ocr_image。
                - 如果 OCR 结果不清楚，可以再调用 ask_vlm 辅助解释。

                4. 通用目标检测
                - 如果用户问图中有哪些常见物体、目标位置、bbox，调用 detect_objects_yolo。
                - YOLO 适合常见类别，不适合任意自定义缺陷。

                5. 开放词汇检测
                - 如果用户问特定对象或缺陷，例如 scratch、crack、stain、logo、screw、defect，而 YOLO 类别可能不包含，调用 grounding_detect。
                - grounding_detect 的 text_prompt 应该使用英文短语，并用句点分隔，例如:
                "scratch . crack . stain . defect ."

                6. 分割 / 区域
                - 如果用户问区域、轮廓、mask、面积、形状，调用 segment_with_sam。
                - SAM 自动分割只能提供候选区域，不能直接证明区域语义，必要时结合 VLM 或 GroundingDINO。

                7. 语义理解
                - 如果用户问整体场景、异常解释、质量判断原因、综合分析，调用 ask_vlm。
                - 不要假装看过图片，必须通过工具获得视觉信息。

                8. 多工具组合
                - 质量 + 内容：detect_blur + ask_vlm
                - 文字 + 内容：ocr_image + ask_vlm
                - 目标位置 + 语义：detect_objects_yolo 或 grounding_detect + ask_vlm
                - 缺陷区域：grounding_detect + segment_with_sam + ask_vlm
                - 不确定结果：说明不确定性，必要时建议人工复核。

                回答要求:
                - 中文回答
                - 明确说明调用了哪些 MCP 工具
                - 区分“当前工具观察”和“历史相似案例”
                - 不要编造工具没有返回的信息
                - 如果工具失败，要说明失败原因并给出下一步建议
                """,
        )

        image_path = state.get("image_path")
        question = state.get("question", "")
        task_type = state.get("task_type", "both")
        retry_count = state.get("retry_count", 0)
        human_feedback = state.get("human_feedback")

        memory_context = state.get("memory_context", {})

        user_message = f"""
            当前任务类型:
            {task_type}

            图片路径:
            {image_path}

            用户问题:
            {question}

            Planner 计划:
            {state.get("plan")}

            重试次数:
            {retry_count}

            人工反馈，如果有:
            {human_feedback}

            可参考的长期记忆:

            same_image_tasks:
            {memory_context.get("same_image_tasks", [])}

            keyword_tasks:
            {memory_context.get("keyword_tasks", [])}

            similar_tasks:
            {memory_context.get("similar_tasks", [])}
            
            similar_images:
            {memory_context.get("similar_images", [])}

            请调用合适的 MCP 工具完成视觉分析。
            如果历史记忆与当前工具结果冲突，以当前工具结果为准，并说明差异。
            不要仅凭历史记忆判断当前图片。
            如果引用相似图片，请明确说“历史相似图片案例显示...”，不要说成当前图片的新观察。
            """

        try:            
            # Graph 有 trace
            # 但 node 内部 create_agent 的调用没有完整 trace
            # 在 node 内部重新创建 handler
            handler = get_langfuse_handler()
            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_message,
                        }
                    ]
                },
                config={
                    "callbacks": [handler],
                    "run_name": "vision_agent_with_mcp_tools",
                    "tags": ["vision-agent", "mcp-tools"],
                },
            )

            # ── 提取最终答案 ──────────────────────────────────────────
            # Qwen3 模型在收到工具结果后，有时会返回 content="" 的
            # AIMessage（间歇性行为：模型认为工具输出已足够，不再总结）。
            # 此时回退到 ToolMessage 的内容拼接。
            answer = result["messages"][-1].content

            if not answer:
                # 从后往前找 ToolMessage，拼接工具返回的内容作为 fallback
                tool_contents: list[str] = []
                for msg in reversed(result["messages"]):
                    if isinstance(msg, ToolMessage) and msg.content:
                        tool_contents.append(str(msg.content))
                    if isinstance(msg, AIMessage) and msg.content:
                        # 遇到有内容的 AIMessage 就停（可能是中间的思考）
                        break
                if tool_contents:
                    tool_contents.reverse()
                    answer = "\n\n".join(tool_contents)
                    answer = (
                        "[注意：模型未生成最终总结，以下为工具直接返回的结果]\n\n"
                        + answer
                    )

            print('debug make vision agent node answer', answer[:200] if answer else '<EMPTY>')
            return {
                "vision_answer": answer if answer else "[Vision Agent 未生成有效回复，请人工检查]",
                "error": None,
            }

        except Exception as e:
            return {
                "error": f"vision_agent failed: {repr(e)}",
                "retry_count": retry_count + 1,
            }

    return vision_agent_node


async def critic_node(state: HybridVisionState) -> dict:
    """
    Critic:
    输出 pass / retry / human_review / fail。
    """

    llm = get_text_llm(temperature=0)

    question = state.get("question", "")
    vision_answer = state.get("vision_answer")
    error = state.get("error")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if error:
        if retry_count < max_retries:
            return {
                "critic_decision": "retry",
                "critic_reason": f"Vision Agent 出错，准备重试。错误: {error}",
            }

        return {
            "critic_decision": "human_review",
            "critic_reason": f"Vision Agent 出错且达到自动重试上限，需要人工判断。错误: {error}",
        }

    system_prompt = """
        你是视觉分析结果审核 Agent。

        你需要判断 Vision Agent 的回答是否足够好，以及是否需要人工复核。

        可选 decision:
        - pass: 回答基本可用
        - retry: 回答缺少关键信息，但可以自动重试
        - human_review: 结果不确定、疑似缺陷、涉及关键判断，应该让人工复核
        - fail: 无法完成任务

        触发 human_review 的典型情况:
        1. 回答中出现“可能”“疑似”“不确定”“看不清”“无法确认”
        2. 涉及缺陷、异常、质量结论但证据不足
        3. 工具结果互相矛盾
        4. 已经达到最大重试次数但还没有可靠结论
        5. 用户显式要求人工确认

        请只输出 JSON，不要输出 Markdown。

        JSON 格式:
        {
        "decision": "pass|retry|human_review|fail",
        "reason": "简短理由"
        }
        """

    memory_context = state.get("memory_context", {})

    user_prompt = f"""
        用户问题:
        {question}

        Vision Agent 回答:
        {vision_answer}

        历史记忆:

        same_image_tasks:
        {memory_context.get("same_image_tasks", [])}

        keyword_tasks:
        {memory_context.get("keyword_tasks", [])}

        similar_tasks:
        {memory_context.get("similar_tasks", [])}
                
        similar_images:
        {memory_context.get("similar_images", [])}

        重试次数:
        {retry_count}

        最大重试次数:
        {max_retries}

        审核标准:
        1. 是否回答了用户问题
        2. 是否说明了工具调用依据
        3. 如果涉及图片内容，是否有 VLM 分析
        4. 如果涉及模糊/清晰度，是否有 blur 检测
        5. 是否存在明显空话、猜测或不确定表达
        6. 当前结论是否与相同图片历史结果明显冲突
        7. 当前结论是否与相似历史案例有明显冲突
        8. 当前结论是否与图像相似历史案例明显冲突
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config={
                "run_name": "critic_llm",
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


def route_after_critic(state: HybridVisionState) -> str:
    decision = state.get("critic_decision")

    if decision == "retry":
        return "increment_retry"

    if decision == "human_review":
        return "human_review"

    return "report"


def increment_retry_node(state: HybridVisionState) -> dict:
    retry_count = state.get("retry_count", 0)
    return {
        "retry_count": retry_count + 1,
    }


def human_review_node(state: HybridVisionState) -> dict:
    """
    Human-in-the-loop 节点。

    这里调用 interrupt() 暂停图执行。
    恢复时，Command(resume=...) 的值会作为 interrupt() 的返回值。
    """

    review_payload = {
        "type": "vision_review_required",
        "message": "需要人工复核视觉分析结果。",
        "question": state.get("question"),
        "image_path": state.get("image_path"),
        "vision_answer": state.get("vision_answer"),
        "critic_decision": state.get("critic_decision"),
        "critic_reason": state.get("critic_reason"),
        "retry_count": state.get("retry_count", 0),
        "observability": {
            "event": "human_review_interrupt",
            "reason": state.get("critic_reason"),
        },
        "allowed_actions": [
            {
                "action": "accept",
                "description": "接受 Vision Agent 的结果，继续生成报告。",
            },
            {
                "action": "edit",
                "description": "人工修改 Vision Agent 的结果，然后生成报告。",
                "fields": ["edited_answer"],
            },
            {
                "action": "retry",
                "description": "要求 Vision Agent 根据人工反馈重新分析。",
                "fields": ["feedback"],
            },
            {
                "action": "reject",
                "description": "拒绝当前分析，直接生成失败报告。",
                "fields": ["feedback"],
            },
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
            "vision_answer": edited_answer,
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


def route_after_human_review(state: HybridVisionState) -> str:
    decision = state.get("human_decision")

    if decision == "retry":
        return "vision_agent"

    return "report"


async def report_node(state: HybridVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    system_prompt = """
        你是视觉分析报告 Agent。

        请根据 Planner、Vision Agent、Critic 和人工复核结果生成最终回答。
                
        如果引用历史记忆，请明确说明“历史记录显示...”，不要把历史记录当成当前图像的新观察结果。
        如果引用 similar_tasks，请明确说明“相似历史案例显示...”，并区分当前工具观察和历史案例。        
        如果引用 similar_images，请明确说明“图像相似历史案例显示...”，并区分：
        1. 当前工具观察
        2. 文本相似历史案例
        3. 图像相似历史案例
        4. 人工复核结果

        要求:
        - 中文
        - 结构清晰
        - 不要编造未被工具或人工确认的信息
        - 如果没有图片，提醒用户提供图片
        - 如果分析失败，说明失败原因和下一步建议
        - 如果人工修改过结果，以人工修改内容为准，并说明经过人工复核
        """

    memory_context = state.get("memory_context", {})

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        图片路径:
        {state.get("image_path")}

        Planner:
        task_type = {state.get("task_type")}
        plan = {state.get("plan")}
        reason = {state.get("planner_reason")}

        Vision Agent 结果:
        {state.get("vision_answer")}

        Critic:
        decision = {state.get("critic_decision")}
        reason = {state.get("critic_reason")}

        Human Review:
        human_decision = {state.get("human_decision")}
        human_feedback = {state.get("human_feedback")}
        human_edited_answer = {state.get("human_edited_answer")}

        长期记忆参考:

        recent_tasks:
        {memory_context.get("recent_tasks", [])}

        same_image_tasks:
        {memory_context.get("same_image_tasks", [])}

        keyword_tasks:
        {memory_context.get("keyword_tasks", [])}

        similar_tasks:
        {memory_context.get("similar_tasks", [])}
        
        similar_images:
        {memory_context.get("similar_images", [])}

        错误:
        {state.get("error")}

        上一轮结果:
        {state.get("last_result")}
        """

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],        
        config={
                "run_name": "report_llm",
                "tags": ["report"],
        },
    )

    return {
        "final_answer": response.content,
    }


def make_save_memory_node(memory_manager):
    def save_memory_node(state: HybridVisionState) -> dict:
        task_id = memory_manager.save_graph_result(state)

        return {
            "task_id": task_id,
        }

    return save_memory_node


def build_hybrid_vision_graph_memory(mcp_tools, memory_manager):
    graph = StateGraph(HybridVisionState)

    graph.add_node("load_memory", make_load_memory_node(memory_manager))
    graph.add_node("planner", planner_node)
    graph.add_node("vision_agent", make_vision_agent_node(mcp_tools))
    graph.add_node("critic", critic_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("report", report_node)
    graph.add_node("save_memory", make_save_memory_node(memory_manager))

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "vision_agent": "vision_agent",
            "report": "report",
        },
    )

    graph.add_edge("vision_agent", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "increment_retry": "increment_retry",
            "human_review": "human_review",
            "report": "report",
        },
    )

    graph.add_edge("increment_retry", "vision_agent")

    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "vision_agent": "vision_agent",
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
            "has_image": str(current_image_path is not None),
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
            "langgraph",
            "mcp",
            "memory",
            "hitl",
        ],
    }

    result = await app.ainvoke(initial_state, config=config)

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        interrupt_value = interrupts[0].value

        print("\n" + "=" * 80)
        print("需要人工复核：")
        print(json.dumps(interrupt_value, ensure_ascii=False, indent=2))
        print("=" * 80)

        print("\n请选择人工动作：")
        print("1. accept  - 接受当前结果")
        print("2. edit    - 修改结果")
        print("3. retry   - 带反馈重试")
        print("4. reject  - 拒绝当前分析")

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
                "feedback": "人工修改了视觉分析结果。",
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

    trace_summary = {
        "task_id": task_id,
        "task_type": result.get("task_type"),
        "critic_decision": result.get("critic_decision"),
        "human_decision": result.get("human_decision"),
        "retry_count": result.get("retry_count"),
        "memory_stats": result.get("memory_stats"),
    }

    print("\nTrace summary:")
    print(json.dumps(trace_summary, ensure_ascii=False, indent=2))

    return True


async def main():
    try:
        mcp_client, mcp_tools = await load_vision_mcp_tools()

        print("Loaded MCP tools:")
        for tool in mcp_tools:
            print(f"- {tool.name}: {tool.description[:120]}\n{'*'*10}")

        session_id = "vision-memory-session-002"

        memory_manager = MemoryManager(
            session_id=session_id,
            db_path="data/memory/vision_memory.sqlite3",
            max_turns=8,
        )

        app = build_hybrid_vision_graph_memory(
            mcp_tools=mcp_tools,
            memory_manager=memory_manager,
        )

        print("\nHybrid Vision Graph Memory v1 started.")
        print("输入 exit 退出。")
        print("第二轮继续分析同一张图片时，Image path 可以留空。")

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
# /home/ziyi/gitlocal/AIDI/test_imgs/WDLD13078D2A_03-Cam1-158-2.bmp
# /home/ziyi/gitlocal/AIDI/test_imgs/WDLD14249F1A_04-Cam2-1150-3.bmp
# /home/ziyi/gitlocal/AIDI/test_imgs/WDLD14439B1A_16-Cam1-1226-3.bmp
# 这张图有没有和以前类似的缺陷
# /home/ziyi/gitlocal/AIDI/test_imgs/WDLD13055F1A_15-Cam1-765-4.bmp
# 请检查图中是否有灰尘、异物，并给出位置。


# 1. OCR:
#    “请读取这张图里的文字。”

# 2. YOLO:
#    “请检测这张图里有哪些常见物体，并返回位置。”

# 3. SAM:
#    “请分割这张图中的主要区域，返回面积最大的几个区域。”

# 4. GroundingDINO:
#    “请检测 scratch . crack . defect . 并给出位置。”

# 5. 多工具综合:
#    “这张图是否模糊？是否有缺陷？如果有，请定位区域并结合历史相似图片说明。”
