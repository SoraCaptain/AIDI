from typing import Any

from langchain.agents import create_agent
from app.agents.local_models import get_text_llm
from app.tools.native_vision_tools import detect_blur, inspect_image
from app.observability.metrics_callback import MetricsCallbackHandler


async def run_quality(state: dict[str, Any]) -> dict:
    tools = [detect_blur, inspect_image]
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
        agent_with_metrics = agent.with_config(
            callbacks=[MetricsCallbackHandler("native")]
        )
        result = await agent_with_metrics.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={
                "run_name": "quality_agent_parallel",
                "tags": ["quality", "parallel", "native-tools"],
            },
        )

        return {"quality_result": result["messages"][-1].content}

    except Exception as e:
        return {"quality_error": repr(e)}
