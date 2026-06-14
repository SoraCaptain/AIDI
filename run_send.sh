
## 创建任务
# curl -X POST http://10.6.88.239:8400/tasks \
#   -F "question=这张图是否模糊？是否存在 scratch 或 crack？如果不确定请进入人工复核。" \
#   -F "file=@/home/ziyi/gitlocal/AIDI/test_imgs/WDED1900240A_04-Cam2-85-1.bmp"


## 查询任务状态
curl http://10.6.88.239:8400/tasks/88cf23d1-4ee4-47e1-8d23-3bbd26cfe1c7

## 提交人工复核：accept
# curl -X POST http://10.6.88.239:8400/tasks/2fdc8052-aeff-48e0-8c7b-91c957799258/human-review \
#   -H "Content-Type: application/json" \
#   -d '{
#     "action": "accept",
#     "feedback": "人工确认当前分析结果可以接受。"
#   }'

## 提交人工复核：edit
# curl -X POST http://10.6.88.239:8400/tasks/88cf23d1-4ee4-47e1-8d23-3bbd26cfe1c7/human-review \
#   -H "Content-Type: application/json" \
#   -d '{
#     "action": "edit",
#     "feedback": "人工修正了缺陷判断。",
#     "edited_answer": "图片存在轻微模糊，不存在 scratch 或 crack。"
#   }'

## 提交人工复核：retry
# curl -X POST http://A800_IP:8400/tasks/{task_id}/human-review \
#   -H "Content-Type: application/json" \
#   -d '{
#     "action": "retry",
#     "feedback": "请重点检查左上角疑似裂纹区域，并结合 GroundingDINO 与 VLM 重新判断。"
#   }'


## 获取报告
# curl http://10.6.88.239:8400/tasks/2fdc8052-aeff-48e0-8c7b-91c957799258/report