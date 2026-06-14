# cv server
conda create -n cv python=3.10 -y
conda activate cv

pip install -r requirements_cv.txt

# grouding dino server
4090D CUDA12.2
gcc --version (9/10/11/12)

conda create -n gdino python=3.10 -y
conda activate gdino
python -m pip install -U pip setuptools wheel

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

pip install numpy==1.26.4
pip install opencv-python==4.8.1.78

验证
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

export CUDA_HOME=/usr/local/cuda-12.2
export TORCH_CUDA_ARCH_LIST="8.9"

验证安装
python - <<'PY'
import torch
import groundingdino
from groundingdino.util.inference import load_model

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "/home/ziyi/gitlocal/AIDI/server/weights/groundingdino_swint_ogc.pth"
)
print("GroundingDINO load ok")
PY



pip uninstall -y transformers tokenizers huggingface-hub

pip install "transformers==4.40.2" \
            "tokenizers==0.19.1" \
            "huggingface-hub>=0.20.0,<1.0"


# Model weights
copy following model into 'weights'
groundingdino_swint_ogc.pth
sam_vit_b_01ec64.pth
yolo11n.pt