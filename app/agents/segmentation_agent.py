from typing import Any

from langchain.agents import create_agent
from app.agents.local_models import get_text_llm
from app.mcp_clients.vision_mcp_client import load_vision_mcp_tools
from .utils import select_tools


async def run_segmentation(state: dict[str, Any]) -> dict:
    _, mcp_tools = await load_vision_mcp_tools()
    tools = select_tools(mcp_tools, ["segment"])
    llm = get_text_llm(temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
            你是 Segmentation Agent，只负责图像分割。

            必须使用 segment_with_sam。
            对图像中的主要区域或物体生成像素级掩码。
            不要做目标检测或文字提取。
            输出中文，描述分割结果中包含的掩码数量和主要区域。
            """,
    )

    user_message = f"""
        图片路径:
        {state.get("image_path")}

        用户问题:
        {state.get("question")}

        Planner 对 Segmentation 的计划:
        {state.get("plan", {}).get("segmentation")}

        已有的上下文结果（可能包含 detection 结果）:
        {state.get("context_results")}
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
