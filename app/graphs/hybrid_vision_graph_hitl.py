# app/graphs/hybrid_vision_graph_hitl.py
import os
import json
import asyncio
from typing import TypedDict, Optional, List, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from app.agents.local_models import get_text_llm
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.memory.session_memory import SessionMemory
from utils.logger import logger

load_dotenv()


class HybridVisionState(TypedDict, total=False):
    question: str
    image_path: Optional[str]

    conversation_history: List[Dict]
    last_result: Optional[str]

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


async def planner_node(state: HybridVisionState) -> dict:
    llm = get_text_llm(temperature=0)

    image_path = state.get("image_path")
    question = state.get("question", "")
    last_result = state.get("last_result")

    system_prompt = """
你是视觉任务规划 Agent。

你只负责判断任务类型，不直接分析图片。

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

    user_prompt = f"""
当前图片路径:
{image_path}

用户问题:
{question}

上一轮分析结果:
{last_result}

最近对话:
{state.get("conversation_history", [])}
"""

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
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
            system_prompt=f"""
                你是一个视觉分析 Agent。

                你通过 MCP tools 调用内部视觉能力。

                当前可用工具:
                {tool_names}

                工具使用规则:
                - 如果用户问清晰度、模糊、质量，必须调用 blur 相关工具。
                - 如果用户问图片内容、缺陷、异常，必须调用 VLM 相关工具。
                - 如果问题同时涉及质量和内容，两个工具都要调用。
                - 不要假装看过图片，必须通过 MCP 工具获得视觉信息。
                - 如果工具失败，说明失败原因。

                回答要求:
                - 中文回答
                - 简洁但有依据
                - 明确说明调用了哪些 MCP 工具
                - 不确定时明确说明不确定性
                """,
            )

        image_path = state.get("image_path")
        question = state.get("question", "")
        task_type = state.get("task_type", "both")
        retry_count = state.get("retry_count", 0)
        human_feedback = state.get("human_feedback")

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

            请调用合适的 MCP 工具完成视觉分析。
            如果这是人工要求的重试，请重点修正人工反馈中指出的问题。
            """

        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_message,
                        }
                    ]
                }
            )

            answer = result["messages"][-1].content
            logger.debug('debug make vision agent node answer %s', answer)
            return {
                "vision_answer": answer,
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

    user_prompt = f"""
用户问题:
{question}

Vision Agent 回答:
{vision_answer}

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
"""

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
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

要求:
- 中文
- 结构清晰
- 不要编造未被工具或人工确认的信息
- 如果没有图片，提醒用户提供图片
- 如果分析失败，说明失败原因和下一步建议
- 如果人工修改过结果，以人工修改内容为准，并说明经过人工复核
"""

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

错误:
{state.get("error")}

上一轮结果:
{state.get("last_result")}
"""

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {
        "final_answer": response.content,
    }


def build_hybrid_vision_graph_hitl(mcp_tools):
    graph = StateGraph(HybridVisionState)

    graph.add_node("planner", planner_node)
    graph.add_node("vision_agent", make_vision_agent_node(mcp_tools))
    graph.add_node("critic", critic_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "planner")

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

    graph.add_edge("report", END)

    checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)


async def run_one_turn(app, memory: SessionMemory, thread_id: str):
    image_path = input("\nImage path, empty if same as before: ").strip()

    if image_path.lower() == "exit":
        return False

    if image_path and os.path.isfile(image_path):
        memory.set_image(image_path)

    question = input("Question: ").strip()

    if question.lower() == "exit":
        return False

    current_image_path = memory.get_image()
    memory.add_message("user", question)

    initial_state = {
        "question": question,
        "image_path": current_image_path,
        "conversation_history": memory.get_messages(),
        "last_result": memory.get_last_result(),
        "retry_count": 0,
        "max_retries": 2,
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = await app.ainvoke(initial_state, config=config)

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        interrupt_value = interrupts[0].value

        logger.info("\n" + "=" * 80)
        logger.info("需要人工复核：")
        logger.info(json.dumps(interrupt_value, ensure_ascii=False, indent=2))
        logger.info("=" * 80)

        logger.info("\n请选择人工动作：")
        logger.info("1. accept  - 接受当前结果")
        logger.info("2. edit    - 修改结果")
        logger.info("3. retry   - 带反馈重试")
        logger.info("4. reject  - 拒绝当前分析")

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

    memory.add_message("assistant", final_answer)
    memory.set_last_result(final_answer)

    logger.info("\n" + "=" * 80)
    logger.info(final_answer)
    logger.info("=" * 80)

    return True


async def main():
    mcp_client, mcp_tools = await load_vision_mcp_tools()

    logger.info("Loaded MCP tools:")
    for tool in mcp_tools:
        logger.info(f"- {tool.name}: {tool.description[:120]}")

    app = build_hybrid_vision_graph_hitl(mcp_tools)
    memory = SessionMemory(max_turns=8)

    logger.info("\nHybrid Vision Graph HITL v1 started.")
    logger.info("输入 exit 退出。")
    logger.info("第二轮继续分析同一张图片时，Image path 可以留空。")

    # 开发阶段固定一个 thread_id。
    # 生产环境应该按 user_id + session_id + task_id 生成。
    thread_id = "vision-hitl-session-001"

    while True:
        should_continue = await run_one_turn(app, memory, thread_id)
        if not should_continue:
            break


if __name__ == "__main__":
    asyncio.run(main())

# /home/ziyi/gitlocal/AIDI/test_imgs/train01.png
# /home/ziyi/gitlocal/AIDI/test_imgs/WDED1900240A_04-Cam2-85-1.bmp