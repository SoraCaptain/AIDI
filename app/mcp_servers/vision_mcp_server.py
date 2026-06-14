# app/mcp_servers/vision_mcp_server.py

import os
import time
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from utils.utils import encode_image

load_dotenv()

CV_SERVER = os.getenv("CV_SERVER", "http://10.6.88.17:8200")
VLM_SERVER = os.getenv("VLM_SERVER", "http://10.6.88.13:8002")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
GDINO_SERVER = os.getenv("GDINO_SERVER", "http://10.6.88.17:8210")

mcp = FastMCP("vision-tools")


def _resolve_image(image_path: str) -> str:
    """If image_path is a local file, encode to base64 data URL so remote
    servers (CV / GDINO) can access it. URLs and existing data: URIs pass
    through unchanged."""
    if image_path.startswith(("http://", "https://", "data:")):
        return image_path
    return encode_image(image_path)


def post_cv(path: str, payload: dict, timeout: int = 60) -> dict:
    resp = requests.post(
        f"{CV_SERVER}{path}",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def post_gdino(path: str, payload: dict, timeout: int = 60) -> dict:
    resp = requests.post(
        f"{GDINO_SERVER}{path}",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def inspect_image(image_path: str) -> dict:
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

    return post_cv(
        "/inspect",
        {"image_path": _resolve_image(image_path)},
        timeout=20,
    )


@mcp.tool()
def detect_blur(image_path: str) -> dict:
    """
    Detect whether an image is blurry.

    Use this tool when the user asks about:
    - blur
    - sharpness
    - clarity
    - focus
    - image quality

    Args:
        image_path: Image path or URL accessible by the CV server.
    """

    return post_cv(
        "/blur",
        {"image_path": _resolve_image(image_path)},
        timeout=20,
    )


@mcp.tool()
def ocr_image(image_path: str) -> dict:
    """
    Extract text from an image using OCR.

    Use this tool when:
    - the user asks what text is in the image
    - the image contains labels, serial numbers, documents, signs, tables, or UI text
    - visual understanding requires reading text

    Args:
        image_path: Image path or URL accessible by the OCR/CV server.
    """

    return post_cv(
        "/ocr",
        {"image_path": _resolve_image(image_path)},
        timeout=120,
    )


@mcp.tool()
def detect_objects_yolo(
    image_path: str,
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict:
    """
    Detect common objects using YOLO.

    Use this tool when:
    - the user asks what objects are present
    - the user asks for bounding boxes
    - the task needs fast closed-set object detection
    - the objects are likely in common YOLO classes

    Args:
        image_path: Image path or URL accessible by the CV server.
        conf: Confidence threshold.
        imgsz: Inference image size.
    """

    return post_cv(
        "/yolo/detect",
        {
            "image_path": _resolve_image(image_path),
            "conf": conf,
            "imgsz": imgsz,
        },
        timeout=60,
    )


@mcp.tool()
def segment_with_sam(
    image_path: str,
    max_masks: int = 20,
) -> dict:
    """
    Automatically segment major regions or objects in an image using SAM.

    Use this tool when:
    - the user asks about regions, masks, contours, or segmentation
    - you need object/region proposals
    - detection boxes are insufficient
    - you need approximate shape/area information

    Args:
        image_path: Image path or URL accessible by the CV server.
        max_masks: Maximum number of mask summaries to return.
    """

    return post_cv(
        "/sam/segment_auto",
        {
            "image_path": _resolve_image(image_path),
            "max_masks": max_masks,
        },
        timeout=180,
    )


@mcp.tool()
def grounding_detect(
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> dict:
    """
    Detect objects or defects by natural language prompt using GroundingDINO.

    Use this tool when:
    - YOLO closed-set classes are insufficient
    - the user asks for a specific object or defect not in YOLO classes
    - the task needs open-vocabulary detection
    - examples: "scratch . crack . defect . logo . screw . stain ."

    Args:
        image_path: Image path or URL accessible by the CV server.
        text_prompt: Text prompt describing objects or defects to detect.
        box_threshold: Box confidence threshold.
        text_threshold: Text matching threshold.
    """

    return post_gdino(
        "/grounding/detect",
        {
            "image_path": _resolve_image(image_path),
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        },
        timeout=180,
    )


@mcp.tool()
def ask_vlm(image_path: str, question: str) -> str:
    """
    Ask the local vision-language model to analyze an image.

    Use this tool when:
    - the user asks about scene understanding
    - the user asks for high-level visual reasoning
    - OCR/detection/segmentation results need semantic explanation
    - you need a natural language visual description

    Args:
        image_path: Image path or URL accessible by the VLM server.
        question: Detailed visual question.

    Returns:
        Natural language answer from the VLM.
    """
    # If it's already a URL or data URL, pass through directly
    if image_path.startswith(("http://", "https://", "data:")):
        image_url = image_path
    else:
        image_url = encode_image(image_path)
    start = time.time()
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
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
        timeout=120,
    )
    resp.raise_for_status()

    answer = resp.json()["choices"][0]["message"]["content"]
    latency_ms = round((time.time() - start) * 1000, 2)

    return (
        f"{answer}\n\n"
        f"[tool_metadata] backend=vlm_server "
        f"model={VLM_MODEL} latency_ms={latency_ms} server={VLM_SERVER}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
