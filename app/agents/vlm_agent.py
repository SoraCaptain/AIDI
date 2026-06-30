from typing import Any

from langchain.agents import create_agent
from app.agents.local_models import get_text_llm
from app.tools.native_vision_tools import vlm_understand_image


async def run_vlm(state: dict[str, Any]) -> dict:
    tools = [vlm_understand_image]
    llm = get_text_llm(temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
            你是 VLM 理解 Agent，只负责多模态视觉理解。

            必须使用 vlm_understand_image。
            对图像进行全面理解：
            - 描述场景内容、物体关系、氛围
            - 进行图像推理和总结
            - 解释 OCR/detection/segmentation 的语义含义

            不要做单独的目标检测或文字提取（已有其他 Agent 负责）。
            输出中文，给出完整的图像分析描述。
            """,
    )

    user_message = f"""
        图片路径:
        {state.get("image_path")}

        用户问题:
        {state.get("question")}

        Planner 对 VLM 的计划:
        {state.get("plan", {}).get("vlm_understanding")}

        已有的上下文结果:
        {state.get("context_results")}
        """

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={
                "run_name": "vlm_agent_parallel",
                "tags": ["vlm", "parallel", "native-tools"],
            },
        )

        return {"vlm_result": result["messages"][-1].content}

    except Exception as e:
        return {"vlm_error": repr(e)}
