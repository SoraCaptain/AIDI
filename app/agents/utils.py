from typing import List


def select_tools(mcp_tools, keywords: List[str]):
    """
    根据工具名关键词筛选工具。
    MCP tool_name_prefix=True 时，工具名通常类似：
    vision_detect_blur
    vision_ocr_image
    vision_grounding_detect
    """
    selected = []

    for tool in mcp_tools:
        name = tool.name.lower()
        if any(k.lower() in name for k in keywords):
            selected.append(tool)

    return selected