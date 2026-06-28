# app/graphs/hybrid_vision_graph.py
import os
import json
from typing import TypedDict, Optional, List, Dict
from dotenv import load_dotenv

from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END

from app.agents.local_models import get_text_llm
from app.tools.langchain_vision_tools import detect_blur, ask_vlm
from app.memory.session_memory import SessionMemory
from utils.logger import logger

load_dotenv()


class HybridVisionState(TypedDict, total=False):
    # 用户输入
    question: str
    image_path: Optional[str]

    # 上下文
    conversation_history: List[Dict]
    last_result: Optional[str]

    # Planner 输出
    plan: Optional[str]
    task_type: Optional[str]
    planner_reason: Optional[str]

    # Vision Agent 输出
    vision_answer: Optional[str]

    # Critic 输出
    critic_decision: Optional[str]
    critic_reason: Optional[str]

    # 控制字段
    retry_count: int
    max_retries: int
    error: Optional[str]

    # 最终输出
    final_answer: Optional[str]
    

def safe_json_loads(text: str) -> dict:
    """
    尽量从模型输出中解析 JSON。
    如果失败，返回空 dict。
    """
    logger.debug("-> log: safe_json_loads")
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


def planner_node(state: HybridVisionState) -> dict:
    """
    Planner Agent:
    判断用户问题属于哪类任务。
    """
    logger.debug("-> log: planner node")
    llm = get_text_llm(temperature=0)

    image_path = state.get("image_path")
    question = state.get("question", "")
    last_result = state.get("last_result")

    system_prompt = """
        你是视觉任务规划 Agent。

        你需要根据用户问题和上下文，判断下一步应该做什么。

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

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    parsed = safe_json_loads(response.content)

    task_type = parsed.get("task_type")

    if not image_path and task_type != "report_only":
        task_type = "no_image"

    if task_type not in [
        "no_image",
        "quality_check",
        "image_understanding",
        "both",
        "report_only",
    ]:
        task_type = "both" if image_path else "no_image"

    return {
        "task_type": task_type,
        "plan": parsed.get("plan", ""),
        "planner_reason": parsed.get("reason", ""),
    }
    

def route_after_planner(state: HybridVisionState) -> str:
    task_type = state.get("task_type")
    logger.debug(f"-> log: route_after_planner {task_type}")
    if task_type == "no_image":
        return "report"

    if task_type == "report_only":
        return "report"

    return "vision_agent"


def vision_agent_node(state: HybridVisionState) -> dict:
    """
    Vision Agent:
    使用 LangChain create_agent，自动调用视觉工具。
    """
    logger.debug("-> log: vision agent node")
    llm = get_text_llm(temperature=0)

    agent = create_agent(
        model=llm,
        tools=[detect_blur, ask_vlm],
        system_prompt="""
            你是一个视觉分析 Agent。

            你可以使用以下工具:
            1. detect_blur: 检查图片是否模糊，返回 blur_score 和 is_blurry
            2. ask_vlm: 调用视觉语言模型理解图片内容

            工具使用规则:
            - 如果用户问清晰度、模糊、质量，必须调用 detect_blur
            - 如果用户问图片内容、缺陷、异常，必须调用 ask_vlm
            - 如果用户问题同时涉及质量和内容，两个工具都要调用
            - 不要假装看过图片，必须通过工具获得视觉信息
            - 如果工具失败，要说明失败原因

            回答要求:
            - 用中文回答
            - 简洁但要有依据
            - 明确说明调用了哪些工具
            - 如果结果不确定，要说明不确定性
            """,
    )

    image_path = state.get("image_path")
    question = state.get("question", "")
    task_type = state.get("task_type", "both")
    retry_count = state.get("retry_count", 0)

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

        请执行视觉分析。
        """

    try:
        result = agent.invoke(
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

        return {
            "vision_answer": answer,
            "error": None,
        }

    except Exception as e:
        return {
            "error": f"vision_agent failed: {repr(e)}",
            "retry_count": retry_count + 1,
        }
        
        
def critic_node(state: HybridVisionState) -> dict:
    """
    Critic:
    判断 Vision Agent 输出是否可用。
    """
    logger.debug("-> log: critic node")
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
                "critic_reason": f"Vision Agent 出错，尝试重试。错误: {error}",
            }
        else:
            return {
                "critic_decision": "fail",
                "critic_reason": f"Vision Agent 出错且达到最大重试次数。错误: {error}",
            }

    system_prompt = """
        你是视觉分析结果审核 Agent。

        你需要判断 Vision Agent 的回答是否足够好。

        可选 decision:
        - pass: 回答基本可用
        - retry: 回答缺少关键信息，需要重新分析
        - fail: 无法完成任务

        请只输出 JSON，不要输出 Markdown。

        JSON 格式:
        {
        "decision": "pass|retry|fail",
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
        5. 是否存在明显空话或无法验证的猜测
        """

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    parsed = safe_json_loads(response.content)

    decision = parsed.get("decision", "pass")
    reason = parsed.get("reason", "")

    if decision == "retry" and retry_count >= max_retries:
        decision = "fail"
        reason = reason + "；但已达到最大重试次数。"

    if decision not in ["pass", "retry", "fail"]:
        decision = "pass"

    return {
        "critic_decision": decision,
        "critic_reason": reason,
    }
    
    
def route_after_critic(state: HybridVisionState) -> str:
    # Critic 后的路由
    
    decision = state.get("critic_decision")
    logger.debug(f"-> log: route_after_critic {decision}")
    if decision == "retry":
        return "vision_agent"

    return "report"


def increment_retry_node(state: HybridVisionState) -> dict:
    retry_count = state.get("retry_count", 0)
    logger.debug(f"-> log: increment_retry_node: {retry_count}")
    return {
        "retry_count": retry_count + 1
    }
    

def report_node(state: HybridVisionState) -> dict:
    """
    Report Agent:
    生成最终用户可读报告。
    """
    logger.debug("-> log: report node")
    llm = get_text_llm(temperature=0)

    system_prompt = """
        你是视觉分析报告 Agent。

        请根据 Planner、Vision Agent、Critic 的结果，生成最终回答。

        要求:
        - 中文
        - 结构清晰
        - 不要编造未被工具验证的信息
        - 如果没有图片，提醒用户提供图片
        - 如果分析失败，说明失败原因和下一步建议
        - 如果结果不确定，明确说明
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

        错误:
        {state.get("error")}

        上一轮结果:
        {state.get("last_result")}
        """

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {
        "final_answer": response.content
    }


'''
START
  ↓
load_context
  ↓
Planner Agent
  ↓
route_by_plan
    ├─ no_image → Report Agent
    ├─ quality_check → Vision Agent
    ├─ image_understanding → Vision Agent
    └─ both → Vision Agent
  ↓
Critic Node
  ↓
route_after_critic
    ├─ retry → Vision Agent
    └─ pass/fail → Report Agent
  ↓
save_context
  ↓
END
'''

    
def build_hybrid_vision_graph():
    graph = StateGraph(HybridVisionState)

    graph.add_node("planner", planner_node)
    graph.add_node("vision_agent", vision_agent_node)
    graph.add_node("critic", critic_node)
    graph.add_node("increment_retry", increment_retry_node)
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
            "vision_agent": "increment_retry",
            "report": "report",
        },
    )

    graph.add_edge("increment_retry", "vision_agent")
    graph.add_edge("report", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_hybrid_vision_graph()
    memory = SessionMemory(max_turns=8)

    logger.info("Hybrid Vision Graph v1 started.")
    logger.info("输入 exit 退出。")
    logger.info("第二轮继续分析同一张图片时，Image path 可以留空。")

    while True:
        image_path = input("\nImage path, empty if same as before: ").strip()

        if image_path.lower() == "exit":
            break

        if image_path and os.path.isfile(image_path):
            memory.set_image(image_path)

        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

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

        result = app.invoke(initial_state)

        final_answer = result.get("final_answer", "")

        memory.add_message("assistant", final_answer)
        memory.set_last_result(final_answer)

        logger.info("\n" + "=" * 80)
        logger.info(final_answer)
        logger.info("=" * 80)


# /home/ziyi/gitlocal/AIDI/test_imgs/blur.png
