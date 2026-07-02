# app/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import time
from functools import wraps
from typing import Dict, Any

# ========== 定义指标 ==========

# 1. 请求计数器（按 Agent 名称、状态分类）
agent_requests = Counter(
    'aidi_agent_requests_total',
    'Total number of agent executions',
    ['agent_name', 'status']  # status: success, failure, fallback
)

# 2. 请求耗时直方图（按 Agent 名称）
agent_duration = Histogram(
    'aidi_agent_duration_seconds',
    'Agent execution duration in seconds',
    ['agent_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # 自定义分桶
)

# 3. 工具调用计数器（按工具名称、模式）
tool_calls = Counter(
    'aidi_tool_calls_total',
    'Total number of tool calls',
    ['tool_name', 'mode']  # mode: native, mcp
)

# 4. 动态路由计划类型计数器
plan_types = Counter(
    'aidi_plan_types_total',
    'Types of execution plans generated',
    ['plan_type']  # 如: ocr_only, detection_only, full, etc.
)

# 5. 当前活跃请求数（用于负载感知）
active_requests = Gauge(
    'aidi_active_requests',
    'Number of active requests currently being processed'
)

# 6. 技能调用计数器
skill_calls = Counter(
    'aidi_skill_calls_total',
    'Total number of skill invocations',
    ['skill_name', 'status']
)

# ========== 装饰器：自动记录指标 ==========

def record_agent_metrics(agent_name: str):
    """
    装饰器，用于自动记录 Agent 执行的耗时和结果
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 增加活跃请求数
            active_requests.inc()
            start = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                # 检查结果中是否有错误标记
                if isinstance(result, dict) and result.get("error"):
                    status = "failure"
                return result
            except Exception:
                status = "failure"
                raise
            finally:
                duration = time.time() - start
                agent_duration.labels(agent_name=agent_name).observe(duration)
                agent_requests.labels(agent_name=agent_name, status=status).inc()
                active_requests.dec()
        return wrapper
    return decorator


def record_tool_call(tool_name: str, mode: str):
    """记录工具调用"""
    tool_calls.labels(tool_name=tool_name, mode=mode).inc()


def record_plan_type(plan_type: str):
    """记录计划类型"""
    plan_types.labels(plan_type=plan_type).inc()


def record_skill_call(skill_name: str, status: str):
    """记录技能调用"""
    skill_calls.labels(skill_name=skill_name, status=status).inc()


def record_agent_execution(agent_name: str, duration: float, status: str):
    """强制记录单次 Agent 执行（非装饰器，支持动态 agent_name）"""
    agent_duration.labels(agent_name=agent_name).observe(duration)
    agent_requests.labels(agent_name=agent_name, status=status).inc()


def get_metrics_snapshot() -> Dict[str, Any]:
    """获取当前所有 Prometheus 指标的摘要，供评估脚本做一致性校验"""
    snapshot = {
        "agent_requests": {},
        "agent_duration_sum": {},
        "agent_duration_count": {},
        "tool_calls": {},
        "plan_types": {},
        "skill_calls": {},
        "active_requests": active_requests._value.get(),
    }
    for metric in [
        agent_requests, tool_calls, plan_types, skill_calls,
    ]:
        for sample in metric.collect()[0].samples:
            if sample.name.endswith("_created"):
                continue
            labels = sample.labels
            value = sample.value
            if metric == agent_requests:
                snapshot["agent_requests"].setdefault(labels["agent_name"], {})[labels["status"]] = value
            elif metric == tool_calls:
                snapshot["tool_calls"].setdefault(labels["tool_name"], {})[labels["mode"]] = value
            elif metric == plan_types:
                snapshot["plan_types"][labels["plan_type"]] = value
            elif metric == skill_calls:
                snapshot["skill_calls"].setdefault(labels["skill_name"], {})[labels["status"]] = value
    for metric in [agent_duration]:
        for sample in metric.collect()[0].samples:
            if sample.name.endswith("_sum"):
                snapshot["agent_duration_sum"][sample.labels["agent_name"]] = sample.value
            elif sample.name.endswith("_count"):
                snapshot["agent_duration_count"][sample.labels["agent_name"]] = sample.value
    return snapshot

# ========== 暴露 metrics 端点（用于 Prometheus 抓取） ==========

async def metrics_endpoint(request):
    """FastAPI 端点，返回 Prometheus 格式的指标"""
    from fastapi import Response
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")
