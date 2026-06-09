# app/tools/langchain_vision_tools.py

import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from .utils import encode_image

load_dotenv()

CV_SERVER = os.getenv("CV_SERVER", "http://localhost:8200")
VLM_SERVER = os.getenv("VLM_SERVER", "http://10.6.88.13:8002")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


@tool
def detect_blur(image_path: str) -> dict:
    """
    Detect whether an image is blurry.

    Use this tool when:
    - the user asks whether an image is blurry
    - the user asks about sharpness, clarity, focus, or image quality
    - the task requires objective blur score

    Input:
        image_path: local path or URL accessible by the CV server

    Returns:
        {
            "blur_score": float,
            "is_blurry": bool
        }
    """

    resp = requests.post(
        f"{CV_SERVER}/blur",
        params={"path": image_path},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


@tool
def ask_vlm(image_path: str, question: str) -> str:
    """
    Ask a vision-language model to analyze an image.

    Use this tool when:
    - the user asks what is in the image
    - the user asks whether there are visible defects or abnormalities
    - the task requires semantic image understanding
    - blur score alone is not sufficient

    Input:
        image_path: image path or URL accessible by the VLM server
        question: detailed visual question

    Returns:
        Natural language answer from the VLM.
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
                        "image_url": {
                            "url": image_url
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
        timeout=60,
    )
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]
