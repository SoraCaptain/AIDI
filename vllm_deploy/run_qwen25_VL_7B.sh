
export CUDA_VISIBLE_DEVICES=0 
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":2,"video":0}'
