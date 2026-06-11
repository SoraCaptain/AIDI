import base64
from pathlib import Path
from PIL import Image
from io import BytesIO


_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def encode_image(image_path: str) -> str:
    """Read a local image file and return a data URL string."""
    ext = Path(image_path).suffix.lower()
    image = Image.open(image_path)
    w, h = image.size
    short = min(w, h)

    if short >= 384:
        # 对于 BMP 等 VLM 不友好的格式，转换为 JPEG
        if ext == ".bmp":
            image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            mime = "image/jpeg"
        else:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = _MIME_MAP.get(ext, "image/jpeg")
    else:
        long_side = max(w, h)
        new_long = int(long_side * 384 / short)
        new_size = (384, new_long) if w < h else (new_long, 384)
        image = image.resize(new_size)
        buffered = BytesIO()
        # 显式指定 format，避免 image.format 为 None 时报错 "unknown file extension: "
        save_format = image.format or "JPEG"
        if save_format == "BMP":
            image = image.convert("RGB")
            save_format = "JPEG"
        image.save(buffered, format=save_format)
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime = _MIME_MAP.get(f".{save_format.lower()}", "image/jpeg")

    return f"data:{mime};base64,{b64}"
