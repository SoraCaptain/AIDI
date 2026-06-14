
## inspect
curl -X POST http://10.6.88.17:8200/inspect \
  -H "Content-Type: application/json" \
  -d '{"image_path":"http://10.6.88.36:9000/tmp/WDLD13078D2A_03-Cam1-158-2.bmp"}'

## blur
curl -X POST http://10.6.88.17:8200/blur \
  -H "Content-Type: application/json" \
  -d '{"image_path":"http://10.6.88.36:9000/tmp/WDLD13078D2A_03-Cam1-158-2.bmp"}'

## ocr
curl -X POST http://10.6.88.17:8200/ocr \
  -H "Content-Type: application/json" \
  -d '{"image_path":"http://10.6.88.36:9000/tmp/WDPD1662230A_30-Cam1-7-2.png"}'

## yolo
curl -X POST http://10.6.88.17:8200/yolo/detect \
  -H "Content-Type: application/json" \
  -d '{"image_path":"http://10.6.88.36:9000/tmp/dog.jpg","conf":0.25,"imgsz":640}'

## sam
curl -X POST http://10.6.88.17:8200/sam/segment_auto \
  -H "Content-Type: application/json" \
  -d '{"image_path":"http://10.6.88.36:9000/tmp/dog.jpg","max_masks":10}'

## grouding dio
curl -X POST http://10.6.88.17:8210/grounding/detect \
  -H "Content-Type: application/json" \
  -d '{"image_path":"http://10.6.88.36:9000/tmp/20250515-151031-097_0514-02neicashangYan_R00_C03_index006.png","text_prompt":"scratch . defect . crack ."}'
