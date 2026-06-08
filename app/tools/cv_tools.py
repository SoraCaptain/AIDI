import requests
from langchain_core.tools import tool

CV_SERVER = "http://localhost:8200"


@tool
def detect_blur(image_path: str) -> dict:
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

    resp = requests.post(
        f"{CV_SERVER}/blur",
        params={"path": image_path},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()