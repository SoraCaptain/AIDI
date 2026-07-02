from typing import Any

from langchain.agents import create_agent
from app.agents.local_models import get_text_llm
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from app.observability.metrics_callback import MetricsCallbackHandler
from .utils import select_tools


async def run_ocr(state: dict[str, Any]) -> dict:
    _, mcp_tools = await load_vision_mcp_tools()
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
        agent_with_metrics = agent.with_config(
            callbacks=[MetricsCallbackHandler("mcp")]
        )
        result = await agent_with_metrics.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={
                "run_name": "ocr_agent_parallel",
                "tags": ["ocr", "parallel", "mcp-tools"],
            },
        )

        return {"ocr_result": result["messages"][-1].content}

    except Exception as e:
        return {"ocr_error": repr(e)}
