# app/api/metrics.py
from fastapi import APIRouter, Response
from app.observability.metrics import generate_latest, REGISTRY

router = APIRouter()

@router.get("/metrics")
async def get_metrics():
    """Prometheus 指标端点"""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")
