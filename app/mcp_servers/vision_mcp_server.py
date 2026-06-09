# app/mcp_servers/vision_mcp_server.py

import os
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

CV_SERVER = os.getenv("CV_SERVER", "http://localhost:8200")
VLM_SERVER = os.getenv("VLM_SERVER", "http://10.6.88.13:8002")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

mcp = FastMCP("vision-tools")


@mcp.tool()
def detect_blur(image_path: str) -> dict:
    """
    Detect whether an image is blurry.

    Use this tool when the user asks about:
    - blur
    - sharpness
    - clarity
    - focus
    - basic image quality

    Args:
        image_path: Image path or URL accessible by the CV server.

    Returns:
        A dictionary with blur_score and is_blurry.
    """

    resp = requests.post(
        f"{CV_SERVER}/blur",
        params={"path": image_path},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def ask_vlm(image_path: str, question: str) -> str:
    """
    Ask the local vision-language model to analyze an image.

    Use this tool when the user asks about:
    - objects in the image
    - scene understanding
    - visible defects
    - abnormalities
    - visual reasoning

    Args:
        image_path: Image path or URL accessible by the VLM server.
        question: Detailed visual question.

    Returns:
        Natural language answer from the VLM.
    """

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path,
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ],
        "temperature": 0,
    }

    resp = requests.post(
        f"{VLM_SERVER}/v1/chat/completions",
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # 本课先用 stdio，最适合本地开发和 LangChain MCP adapter。
    mcp.run(transport="stdio")
