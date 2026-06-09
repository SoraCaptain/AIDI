# app/tools/cv_api.py

import os
import requests


CV_SERVER = os.getenv("CV_SERVER", "http://localhost:8200")


def detect_blur_api(image_path: str) -> dict:
    """
    调用 4090D 上的 CV Server 检查图片是否模糊。
    """
    resp = requests.post(
        f"{CV_SERVER}/blur",
        params={"path": image_path},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
