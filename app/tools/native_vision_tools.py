# app/tools/native_vision_tools.py
"""
原生视觉工具（绕过 MCP，直接调用后端服务）
使用 @tool 装饰器，延迟更低，更易调试
"""
import asyncio
import base64
import json
from typing import Optional, List, Dict, Any
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import settings
from utils.utils import encode_image

# ---------- 请求/响应模型（便于类型提示） ----------
class DetectionResult(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    scores: List[float]


class SegmentationResult(BaseModel):
    masks: List[str]  # base64 编码的掩码
    labels: List[str]
    scores: List[float]


# ---------- 核心工具函数 ----------

@tool
async def detect_objects(image_path: str, threshold: float = 0.3) -> str:
    """
    使用 CV Server 检测图像中的物体。
    
    Args:
        image_path: 图像的本地路径或 HTTP URL
        threshold: 检测置信度阈值，默认 0.3
    
    Returns:
        JSON 字符串，包含检测到的 boxes, labels, scores
    """
    async with httpx.AsyncClient(timeout=settings.cv_timeout) as client:
        response = await client.post(
            f"{settings.cv_server}/detect",
            json={"image_path": image_path, "threshold": threshold}
        )
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, ensure_ascii=False, indent=2)


@tool
async def segment_objects(image_path: str, threshold: float = 0.3) -> str:
    """
    使用 CV Server 对图像进行实例分割。
    
    Args:
        image_path: 图像的本地路径或 HTTP URL
        threshold: 分割置信度阈值，默认 0.3
    
    Returns:
        JSON 字符串，包含 masks (base64), labels, scores
    """
    async with httpx.AsyncClient(timeout=settings.cv_timeout) as client:
        response = await client.post(
            f"{settings.cv_server}/segment",
            json={"image_path": image_path, "threshold": threshold}
        )
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, ensure_ascii=False, indent=2)


@tool
async def grounding_dino(image_path: str, text_prompt: str, threshold: float = 0.3) -> str:
    """
    使用 GroundingDINO Server 根据文本描述在图像中定位目标。
    
    Args:
        image_path: 图像的本地路径或 HTTP URL
        text_prompt: 文本描述，如 "a red car"
        threshold: 置信度阈值，默认 0.3
    
    Returns:
        JSON 字符串，包含检测到的 boxes 和对应文本
    """
    async with httpx.AsyncClient(timeout=settings.gdino_timeout) as client:
        response = await client.post(
            f"{settings.gdino_server}/grounding",
            json={"image_path": image_path, "text_prompt": text_prompt, "threshold": threshold}
        )
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, ensure_ascii=False, indent=2)


@tool
async def ocr_image(image_path: str) -> str:
    """
    使用 CV Server 对图像进行 OCR 文字识别。
    
    Args:
        image_path: 图像的本地路径或 HTTP URL
    
    Returns:
        识别出的文字内容（JSON 格式，包含文本和位置）
    """
    async with httpx.AsyncClient(timeout=settings.cv_timeout) as client:
        response = await client.post(
            f"{settings.cv_server}/ocr",
            json={"image_path": image_path}
        )
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, ensure_ascii=False, indent=2)


def _resolve_image(image_path: str) -> str:
    """If image_path is a local file, encode to base64 data URL so remote
    servers (CV / GDINO) can access it. URLs and existing data: URIs pass
    through unchanged."""
    if image_path.startswith(("http://", "https://", "data:")):
        return image_path
    return encode_image(image_path)


@tool
async def vlm_understand_image(image_path: str, question: str) -> str:
    """
    使用 VLM (Qwen3-VL) 理解图像内容并回答问题。
    
    Args:
        image_path: 图像的本地路径或 HTTP URL
        question: 关于图像的问题，如 "这张图里有什么？"
    
    Returns:
        VLM 生成的文本回答
    """
    payload = {
        "model": settings.vlm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _resolve_image(image_path)}},
                    {"type": "text", "text": question}
                ]
            }
        ],
        "max_tokens": 512,
    }
    async with httpx.AsyncClient(timeout=settings.vlm_timeout) as client:
        # 这是 异步 HTTP 客户端，配合 async/await 使用：
        # AsyncClient：创建一个支持异步的 HTTP 客户端
        # async with：上下文管理器（自动关闭连接）
        # await client.post(...)：非阻塞发送请求
        # 关键点：不会阻塞当前线程
        # 它会自动帮你：
        # 打开连接池
        # 在退出时关闭 client（释放资源）
        # 避免连接泄漏
        response = await client.post(
            f"{settings.vlm_server}/v1/chat/completions",  # 假设兼容 OpenAI 格式
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", str(data))
            return f"VLM API error (status={response.status_code}): {error_msg}"
        if "choices" not in data:
            # Debug: return the full response so we can see what the server actually returned
            return f"VLM returned unexpected format (no 'choices' key). Full response: {data}"
        return data["choices"][0]["message"]["content"]


@tool
async def detect_blur(image_path: str) -> dict:
    """
    Detect whether an image is blurry.

    Use this tool when:
    - the user asks about image quality
    - blur / sharpness / clarity is mentioned

    Returns:
        {
          "blur_score": float,
          "is_blurry": bool
        }
    """
    async with httpx.AsyncClient(timeout=settings.cv_timeout) as client:
        response = await client.post(
            f"{settings.cv_server}/blur",
            json={"image_path": _resolve_image(image_path)}
        )
        response.raise_for_status()
        return response.json()


@tool()
async def inspect_image(image_path: str) -> dict:
    """
    Inspect image metadata.

    Use this tool when you need basic image information:
    - width
    - height
    - mode
    - whether the image path or URL is accessible

    Args:
        image_path: Image path or URL accessible by the CV server.
    """
    async with httpx.AsyncClient(timeout=settings.cv_timeout) as client:
        response = await client.post(
            f"{settings.cv_server}/inspect",
            json={"image_path": image_path}
        )
        response.raise_for_status()
        return response.json()
    
    
# ---------- 工具列表（供注册中心使用） ----------
NATIVE_VISION_TOOLS = [
    vlm_understand_image,
    detect_blur,
]

# 工具名称映射（方便去重和调试）
NATIVE_TOOL_NAMES = {tool.name for tool in NATIVE_VISION_TOOLS}
