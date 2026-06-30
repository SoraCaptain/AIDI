# app/graphs/dynamic_vision_graph.py
"""
Dynamic Vision Graph — 使用 DynamicRouter 进行 Agent 调度

架构:
- planner → DynamicRouter.generate_plan() 生成执行计划
- router  → DynamicRouter.execute_plan() 执行各 agent
- aggregator → 汇总结果生成最终答案
"""

import asyncio
import json
import os
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

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


def build_dynamic_vision_graph(checkpointer, llm, tool_registry: ToolRegistry):
    """构建使用 DynamicRouter 的视觉分析图。"""
    graph = StateGraph(VisionGraphState)

    router = DynamicRouter(llm, tool_registry)

    async def planner_node(state: dict[str, Any]) -> dict[str, Any]:
        plan = await router.generate_plan(state)
        return {"execution_plan": plan.model_dump()}

    async def router_node(state: dict[str, Any]) -> dict[str, Any]:
        plan_dict = state.get("execution_plan")
        if not plan_dict:
            return {"agent_results": {}, "final_answer": "无法生成执行计划。"}
        results = await router.execute_plan(state, plan_dict)
        return {"agent_results": results}

    async def aggregator_node(state: dict[str, Any]) -> dict[str, Any]:
        results = state.get("agent_results", {})
        parts = ["分析结果："]
        for name, res in results.items():
            if isinstance(res, dict) and "error" in res:
                parts.append(f"- {name}: 失败 ({res['error']})")
            else:
                parts.append(f"- {name}: 完成")
        final_answer = "\n".join(parts)

        # 如果有 VLM 结果，优先作为最终回答
        if "vlm_understanding" in results and isinstance(results["vlm_understanding"], dict):
            vlm_data = results["vlm_understanding"]
            text = vlm_data.get("vlm_result") or vlm_data.get("text")
            if text:
                final_answer = text

        return {"final_answer": final_answer}

    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("aggregator", aggregator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile(checkpointer=checkpointer)


async def run_one_turn(app, memory_manager: MemoryManager, thread_id: str) -> bool:
    """单轮交互：读取用户输入，调用 graph，输出结果。"""
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
        "tags": ["vision-agent", "dynamic-router", "langgraph", "memory"],
    }

    result = await app.ainvoke(initial_state, config=config)

    final_answer = result.get("final_answer", "")
    memory_manager.add_assistant_message(final_answer)

    task_id = result.get("task_id")
    logger.info("\n" + "=" * 80)
    logger.info(final_answer)
    if task_id:
        logger.info(f"\n保存到长期记忆 task_id: {task_id}")
    logger.info("=" * 80)

    logger.info(
        "Trace summary:\n"
        + json.dumps(
            {
                "task_id": task_id,
                "final_answer_preview": final_answer[:200] if final_answer else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return True


async def main():
    """入口：初始化各组件并启动交互式对话循环。"""
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

        checkpointer = InMemorySaver()

        tool_registry = ToolRegistry()
        await tool_registry.initialize()

        llm = get_text_llm(temperature=0)

        app = build_dynamic_vision_graph(
            checkpointer=checkpointer,
            llm=llm,
            tool_registry=tool_registry,
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
        await tool_registry.close()
        flush_langfuse()


if __name__ == "__main__":
    asyncio.run(main())
