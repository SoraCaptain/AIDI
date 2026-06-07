export CUDA_VISIBLE_DEVICES=0
nohup vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8003 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-uvicorn-access-log \
  --uvicorn-log-level warning \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > vllm_qwen3-30B-A3B-Instruct-2507.log 2>&1 &
