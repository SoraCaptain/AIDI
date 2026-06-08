import base64
from pathlib import Path

import requests
from langchain_core.tools import tool

VLM_SERVER = "http://10.6.88.13:8002"

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _encode_image(image_path: str) -> str:
    """Read a local image file and return a data URL string."""
    ext = Path(image_path).suffix.lower()
    mime = _MIME_MAP.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@tool
def ask_vlm(image_path: str, question: str) -> str:
    """
    Ask a vision-language model about an image.

    Use when:
    - user asks about objects, defects, scene understanding
    - CV tools cannot answer

    Input:
      image_path: path or URL
      question: detailed question

    Returns:
      string answer
    """

    # If it's a local path (not a URL), encode to base64 data URL
    if image_path.startswith(("http://", "https://", "data:")):
        image_url = image_path
    else:
        image_url = _encode_image(image_path)

    payload = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    }

    resp = requests.post(
        f"{VLM_SERVER}/v1/chat/completions",
        json=payload,
        timeout=60
    )

    data = resp.json()
    if resp.status_code != 200:
        error_msg = data.get("error", {}).get("message", str(data))
        return f"VLM API error (status={resp.status_code}): {error_msg}"
    if "choices" not in data:
        # Debug: return the full response so we can see what the server actually returned
        return f"VLM returned unexpected format (no 'choices' key). Full response: {data}"

    return data["choices"][0]["message"]["content"]