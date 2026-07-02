from typing import Any

from langchain.agents import create_agent
from app.agents.local_models import get_text_llm
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.observability.metrics_callback import MetricsCallbackHandler
from .utils import select_tools


async def run_grounding_dino(state: dict[str, Any]) -> dict:
    _, mcp_tools = await load_vision_mcp_tools()
    tools = select_tools(mcp_tools, ["grounding"])
    llm = get_text_llm(temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
            你是 GroundingDINO Agent，只负责根据文本描述定位特定物体。

            必须使用 grounding_detect。
            根据用户提供的文本描述，在图像中精确找到对应物体的位置。
            文本提示（text_prompt）应从用户问题中提取或推断。
            不要做 OCR 或通用检测。
            输出中文，包含检测到的物体的位置（bounding box）和匹配文本。
            """,
    )

    user_message = f"""
        图片路径:
        {state.get("image_path")}

        用户问题:
        {state.get("question")}

        Planner 对 GroundingDINO 的计划:
        {state.get("plan", {}).get("grounding_dino")}

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
                "run_name": "grounding_dino_agent_parallel",
                "tags": ["grounding_dino", "parallel", "mcp-tools"],
            },
        )

        return {"grounding_dino_result": result["messages"][-1].content}

    except Exception as e:
        return {"grounding_dino_error": repr(e)}
