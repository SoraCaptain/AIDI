from typing import Any

from langchain.agents import create_agent
from app.agents.local_models import get_text_llm
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.observability.metrics_callback import MetricsCallbackHandler
from .utils import select_tools


async def run_detection(state: dict[str, Any]) -> dict:
    _, mcp_tools = await load_vision_mcp_tools()
    tools = select_tools(mcp_tools, ["detect"])
    llm = get_text_llm(temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
            你是 Detection Agent，只负责目标检测。

            必须使用 detect_objects_yolo。
            识别图像中的物体，返回类别、位置和置信度。
            不要做文字提取或缺陷判断。
            输出中文，包含所有检测到的物体列表和置信度。
            """,
    )

    user_message = f"""
        图片路径:
        {state.get("image_path")}

        用户问题:
        {state.get("question")}

        Planner 对 Detection 的计划:
        {state.get("plan", {}).get("detection")}

        已有的上下文结果:
        {state.get("context_results")}
        """

    try:
        agent_with_metrics = agent.with_config(
            callbacks=[MetricsCallbackHandler("mcp")]
        )
        result = await agent_with_metrics.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={
                "run_name": "detection_agent_parallel",
                "tags": ["detection", "parallel", "mcp-tools"],
            },
        )

        return {"detection_result": result["messages"][-1].content}

    except Exception as e:
        return {"detection_error": repr(e)}
