# app/tools/vlm_api.py

import os
import requests
from utils.utils import encode_image

VLM_SERVER = os.getenv("VLM_SERVER", "http://10.6.88.13:8002")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


def ask_vlm_api(image_path: str, question: str) -> str:
    """
    调用 L20 上的 VLM Server 分析图片。
    image_path 可以是 VLM 服务可访问的本地路径或 URL。
    """

    # If it's a local path (not a URL), encode to base64 data URL
    if image_path.startswith(("http://", "https://", "data:")):
        image_url = image_path
    else:
        image_url = encode_image(image_path)
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
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
        timeout=60,
    )
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]