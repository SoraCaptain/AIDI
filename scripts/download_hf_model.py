#模型下载
# from modelscope import snapshot_download
# save_dir = '/srv/new_storage/ziyi/other/pretrained_models/dinov3/dinov3-vitl16-pretrain-lvd1689m'
# model_dir = snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m', local_dir=save_dir)

# pip install -U huggingface_hub
from huggingface_hub import snapshot_download
import os

from utils.logger import logger

# 如果你想把 token 写死，可改为 token="hf_xxx"
local_dir = "/srv/new_storage/ziyi/other/pretrained_models/huggingface/Ultralytics/YOLO11"
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id="Ultralytics/YOLO11",
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 纯文件，不建软链
    resume_download=True,
    max_workers=8
)

logger.info("全部文件已下载到：" + os.path.abspath(local_dir))
