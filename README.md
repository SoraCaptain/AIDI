
## start cv server 
cd server
uv run uvicorn cv_server:app --host 0.0.0.0 --port 8200


## file transfer
cd /mnt/shared
python -m http.server 9000
image url: http://IP:9000/images/test.jpg


## tools
MCP:
    ocr_image
    detect_objects_yolo
    segment_with_sam
    grounding_detect
Native tools:
    vlm_understand_image,
    detect_blur,
    inspect_image
