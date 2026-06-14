export CV_DEVICE=cuda
export YOLO_MODEL_PATH=weights/yolo11n.pt

# 如果启用 SAM
export SAM_CHECKPOINT=weights/sam_vit_b_01ec64.pth
export SAM_MODEL_TYPE=vit_b

# Disable oneDNN/MKLDNN to avoid PIR attribute conversion errors with PaddlePaddle 3.x
# Ref: NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]
export FLAGS_use_mkldnn=0
export FLAGS_enable_pir_api=0

uvicorn cv_server:app --host 0.0.0.0 --port 8200 > log_cv.log 2>&1 &

# lsof -i :8200