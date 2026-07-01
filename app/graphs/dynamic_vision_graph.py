# app/graphs/dynamic_vision_graph.py
"""
Dynamic Vision Graph — 使用 DynamicRouter 进行 Agent 调度

架构:
- load_memory → planner → DynamicRouter.generate_plan() 生成执行计划
- router → DynamicRouter.execute_plan() 执行各 agent
- aggregator → critic → (prepare_retry / human_review / report) → report → save_memory
"""

import asyncio
import json
import os
from typing import Any, List

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
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
from app.graphs.dynamic_router import DynamicRouter
from app.graphs.state import VisionGraphState
from app.tools.tool_registry import ToolRegistry
from utils.logger import logger

load_dotenv()


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


def make_load_memory_node(memory_manager: MemoryManager):
    def load_memory_node(state: VisionGraphState) -> dict:
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


async def critic_node(state: VisionGraphState) -> dict:
    llm = get_text_llm(temperature=0)

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    system_prompt = """
        你是 Critic Agent。

        你需要审核动态多 Agent 汇总结果是否足够可靠。

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

    agent_results = state.get("agent_results", {})

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        Aggregated Result:
        {state.get("aggregated_result")}

        Agent Results:
        {json.dumps(agent_results, ensure_ascii=False)}

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
            "run_name": "critic_agent_dynamic",
            "tags": ["critic", "dynamic"],
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


def human_review_node(state: VisionGraphState) -> dict:
    agent_results = state.get("agent_results", {})

    review_payload = {
        "type": "dynamic_vision_review_required",
        "message": "需要人工复核动态多 Agent 视觉分析结果。",
        "question": state.get("question"),
        "image_path": state.get("image_path"),
        "aggregated_result": state.get("aggregated_result"),
        "agent_results": agent_results,
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


async def report_node(state: VisionGraphState) -> dict:
    llm = get_text_llm(temperature=0)

    system_prompt = """
        你是 Report Agent。

        请根据动态多 Agent 结果生成最终视觉分析报告。

        要求:
        - 中文
        - 结构清晰
        - 区分当前工具观察、历史相似案例、人工复核
        - 不要编造未被工具或人工确认的信息
        - 如果有人工 edit，以人工 edit 为准
        - 如果分析失败，说明失败原因和下一步建议
        """

    agent_results = state.get("agent_results", {})

    user_prompt = f"""
        用户问题:
        {state.get("question")}

        图片:
        {state.get("image_path")}

        Aggregated:
        {state.get("aggregated_result")}

        Agent Results:
        {json.dumps(agent_results, ensure_ascii=False)}

        Memory:
        memory_result = {state.get("memory_result")}

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
            "run_name": "report_agent_dynamic",
            "tags": ["report", "dynamic"],
        },
    )

    return {"final_answer": response.content}


def prepare_retry_node(state: VisionGraphState) -> dict:
    return {
        "agent_results": None,
        "aggregated_result": None,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def route_after_critic(state: VisionGraphState) -> str:
    decision = state.get("critic_decision")

    if decision == "retry":
        return "prepare_retry"

    if decision == "human_review":
        return "human_review"

    return "report"


def route_after_human_review(state: VisionGraphState) -> str:
    if state.get("human_decision") == "retry":
        return "prepare_retry"

    return "report"


def make_save_memory_node(memory_manager: MemoryManager):
    def save_memory_node(state: VisionGraphState) -> dict:
        agent_results = state.get("agent_results", {})

        save_state = dict(state)

        save_state["vision_answer"] = state.get("aggregated_result")
        save_state["task_type"] = ",".join(agent_results.keys()) if agent_results else "dynamic"

        task_id = memory_manager.save_graph_result(save_state)

        return {"task_id": task_id}

    return save_memory_node


async def build_dynamic_vision_graph(
    mcp_tools,
    memory_manager: MemoryManager,
    checkpointer=None,
):
    graph = StateGraph(VisionGraphState)

    llm = get_text_llm(temperature=0)
    tool_registry = ToolRegistry()
    await tool_registry.initialize()
    router = DynamicRouter(llm, tool_registry)

    async def planner_node(state: VisionGraphState) -> dict:
        plan = await router.generate_plan(state)
        return {"execution_plan": plan.model_dump()}

    async def router_node(state: VisionGraphState) -> dict:
        plan_dict = state.get("execution_plan")
        if not plan_dict:
            return {"agent_results": {}, "aggregated_result": "无法生成执行计划。"}
        results = await router.execute_plan(state, plan_dict)
        return {"agent_results": results}

    async def aggregator_node(state: VisionGraphState) -> dict:
        llm_agg = get_text_llm(temperature=0)

        results = state.get("agent_results", {})

        system_prompt = """
            你是 Aggregator，负责把动态 Router 调度出的 Agent 输出汇总成中间分析结果。

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

            Agent Results:
            {json.dumps(results, ensure_ascii=False, indent=2)}

            Memory:
            memory_result = {state.get("memory_result")}
            memory_error = {state.get("memory_error")}
            """

        response = await llm_agg.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            config={
                "run_name": "dynamic_aggregator",
                "tags": ["aggregator", "dynamic"],
            },
        )

        return {"aggregated_result": response.content}

    graph.add_node("load_memory", make_load_memory_node(memory_manager))
    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_node("critic", critic_node)
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("report", report_node)
    graph.add_node("save_memory", make_save_memory_node(memory_manager))

    graph.add_edge("load_memory", "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "aggregator")
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

    graph.add_edge("prepare_retry", "planner")

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

    graph.set_entry_point("load_memory")

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)


async def run_one_turn(app, memory_manager: MemoryManager, thread_id: str) -> bool:
    image_path = input("\nImage path, empty if same as before: ").strip()
    if image_path.lower() == "exit":
        return False

    while image_path and not os.path.isfile(image_path):
        image_path = input("Valid image path, enter again: ").strip()
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
        extra={"graph": "dynamic_vision_graph"},
    )

    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": trace_metadata,
        "tags": ["vision-agent", "dynamic-router", "langgraph", "memory", "hitl"],
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
    if task_id:
        logger.info(f"\n保存到长期记忆 task_id: {task_id}")
    logger.info("=" * 80)

    logger.info("\nTrace summary:")
    logger.info(
        json.dumps(
            {
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

        session_id = "dynamic-vision-session-001"

        memory_manager = MemoryManager(
            session_id=session_id,
            db_path="data/memory/vision_memory.sqlite3",
            max_turns=8,
            enable_vector_memory=True,
            enable_image_vector_memory=True,
        )

        app = await build_dynamic_vision_graph(
            mcp_tools=mcp_tools,
            memory_manager=memory_manager,
        )

        logger.info("\nDynamic Vision Graph started.")
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

    except KeyboardInterrupt:
        logger.info("\n\n已退出。")
    finally:
        flush_langfuse()


if __name__ == "__main__":
    asyncio.run(main())
