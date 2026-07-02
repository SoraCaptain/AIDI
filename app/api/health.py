# app/api/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.get("/ready")
async def readiness_check():
    # 可以检查 PostgreSQL 连接、MCP 工具等是否就绪
    return {"status": "ready"}