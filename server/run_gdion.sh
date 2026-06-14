export CUDA_VISIBLE_DEVICES=3

export GROUNDING_DINO_CONFIG=/home/ziyi/gitlocal/AIDI/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export GROUNDING_DINO_CHECKPOINT=/home/ziyi/gitlocal/AIDI/server/weights/groundingdino_swint_ogc.pth

uvicorn grounding_server:app --host 0.0.0.0 --port 8210 > log_gdino.log 2>&1 &