
export $(cat .env | xargs) 
uv run uvicorn app.api.gateway:app --host 0.0.0.0 --port 8400 > log_backend.log 2>&1 &