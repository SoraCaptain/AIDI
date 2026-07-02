# tests/evaluation/evaluate.py
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from app.services.graph_runtime_persistent import PersistentGraphRuntime
from app.config import settings
from app.observability.metrics import get_metrics_snapshot
from app.observability.langfuse_client import flush_langfuse

TEST_SET = [
    {
        "id": "test_001",
        "image": "test_imgs/Family.jpg",
        "question": "图片中有几个人？",
        "expected_agents": ["detection"],
        "expected_answer_contains": ["4", "四"]
    },
    # {
    #     "id": "test_002",
    #     "image": "test_imgs/menu.png",
    #     "question": "提取文字",
    #     "expected_agents": ["ocr"],
    #     "expected_answer_contains": ["Function", "and", "material", "of", "each", "part"]
    # },
]

class Evaluator:
    def __init__(self, runtime: PersistentGraphRuntime):
        self.runtime = runtime
        self.results = []

    async def run_single_test(self, test_case: Dict) -> Dict:
        """运行单个测试用例"""
        task_id = f"eval_{test_case['id']}"
        thread_id = f"eval_thread_{test_case['id']}"
        
        start = time.time()
        try:
            result = await self.runtime.run_task(
                task_id=task_id,
                thread_id=thread_id,
                question=test_case["question"],
                image_url=test_case["image"]
            )
            duration = time.time() - start
            final_answer = result.get("final_answer") or ""
            status = result.get("status", "failed")
            
            executed_agents = result.get("trace_summary", {}).get("required_agents", [])
            
            expected = test_case.get("expected_answer_contains", [])
            contains_expected = all(exp in final_answer for exp in expected)
            
            expected_agents = set(test_case.get("expected_agents", []))
            actual_agents = set(executed_agents)
            agent_match = expected_agents == actual_agents
            
            return {
                "id": test_case["id"],
                "success": status == "completed",
                "duration": duration,
                "final_answer": final_answer,
                "contains_expected": contains_expected,
                "agent_match": agent_match,
                "executed_agents": executed_agents,
                "expected_agents": list(expected_agents),
                "error": None
            }
        except Exception as e:
            return {
                "id": test_case["id"],
                "success": False,
                "duration": time.time() - start,
                "error": str(e)
            }

    async def run_all(self, test_set: List[Dict] = None):
        """运行所有测试"""
        test_set = test_set or TEST_SET
        self.results = []
        for case in test_set:
            print(f"🔄 运行测试 {case['id']}...")
            res = await self.run_single_test(case)
            self.results.append(res)
            print(f"   {'✅' if res['success'] else '❌'} 完成, 耗时 {res['duration']:.2f}s")
        return self.results

    def generate_report(self) -> str:
        """生成评测报告"""
        df = pd.DataFrame(self.results)
        total = len(df)
        success_count = df['success'].sum() if 'success' in df else 0
        success_rate = (success_count / total) * 100
        avg_duration = df['duration'].mean() if 'duration' in df else 0
        
        report = f"""
            # 评测报告
            生成时间: {datetime.now()}

            ## 总体指标
            - 总用例数: {total}
            - 成功率: {success_rate:.1f}%
            - 平均耗时: {avg_duration:.2f}s

            ## 详细结果
            {df.to_markdown()}

            ## 失败用例
            {df[df['success'] == False][['id', 'error']].to_markdown() if 'error' in df else '无'}
            """
        return report

    def verify_metrics(self, baseline_snapshot: Dict) -> Dict[str, bool]:
        """读取 Prometheus 指标并与评估统计做一致性校验，验证 metrics.py 是否生效"""
        after_snapshot = get_metrics_snapshot()
        checks = {}
        df = pd.DataFrame(self.results)

        # ------- 1. agent_requests 校验 -------
        baseline_agents = baseline_snapshot.get("agent_requests", {})
        after_agents = after_snapshot.get("agent_requests", {})
        delta_agents = {
            agent: {
                status: after_agents.get(agent, {}).get(status, 0) - baseline_agents.get(agent, {}).get(status, 0)
                for status in set(
                    list(after_agents.get(agent, {}).keys()) +
                    list(baseline_agents.get(agent, {}).keys())
                )
            }
            for agent in set(list(after_agents.keys()) + list(baseline_agents.keys()))
        }
        total_prom_requests = sum(
            delta_agents[agent].get(status, 0) for agent in delta_agents for status in delta_agents.get(agent, {})
        )
        total_eval_requests = len(df)
        checks["agent_requests_count"] = total_prom_requests > 0

        # ------- 2. agent_duration 校验 -------
        after_dur_count = after_snapshot.get("agent_duration_count", {})
        baseline_dur_count = baseline_snapshot.get("agent_duration_count", {})
        delta_dur = {k: after_dur_count.get(k, 0) - baseline_dur_count.get(k, 0) for k in set(list(after_dur_count.keys()) + list(baseline_dur_count.keys()))}
        checks["agent_duration_recorded"] = any(v > 0 for v in delta_dur.values())

        # ------- 3. tool_calls 校验 -------
        after_tools = after_snapshot.get("tool_calls", {})
        baseline_tools = baseline_snapshot.get("tool_calls", {})
        delta_tools = {
            tool: {
                mode: after_tools.get(tool, {}).get(mode, 0) - baseline_tools.get(tool, {}).get(mode, 0)
                for mode in set(list(after_tools.get(tool, {}).keys()) + list(baseline_tools.get(tool, {}).keys()))
            }
            for tool in set(list(after_tools.keys()) + list(baseline_tools.keys()))
        }
        total_tool_calls = sum(delta_tools[tool].get(mode, 0) for tool in delta_tools for mode in delta_tools.get(tool, {}))
        checks["tool_calls_recorded"] = total_tool_calls > 0

        # ------- 4. plan_types 校验 -------
        after_plans = after_snapshot.get("plan_types", {})
        baseline_plans = baseline_snapshot.get("plan_types", {})
        delta_plans = {k: after_plans.get(k, 0) - baseline_plans.get(k, 0) for k in set(list(after_plans.keys()) + list(baseline_plans.keys()))}
        total_plans = sum(v for v in delta_plans.values())
        checks["plan_types_recorded"] = total_plans > 0

        # ------- 5. 打印对比摘要 -------
        print("\n" + "=" * 60)
        print("📊 Prometheus 指标校验结果 (metrics.py 一致性验证)")
        print("=" * 60)
        print(f"  评估用例数:          {total_eval_requests}")
        print(f"  Prometheus agent_requests 增量:  {total_prom_requests}  |  {'✅ 已记录' if checks['agent_requests_count'] else '❌ 未记录'}")
        print(f"  Prometheus agent_duration 增量:  {sum(delta_dur.values())}  |  {'✅ 已记录' if checks['agent_duration_recorded'] else '❌ 未记录'}")
        print(f"  Prometheus tool_calls 增量:       {total_tool_calls}  |  {'✅ 已记录' if checks['tool_calls_recorded'] else '⚠️  无工具调用（无外部服务？）'}")
        print(f"  Prometheus plan_types 增量:       {total_plans}  |  {'✅ 已记录' if checks['plan_types_recorded'] else '⚠️  无计划记录（未用 dynamic_router？）'}")
        if delta_agents:
            print(f"\n  Agent 请求分布:")
            for agent, counts in sorted(delta_agents.items()):
                for status, cnt in sorted(counts.items()):
                    if cnt > 0:
                        print(f"    {agent}: {status}={int(cnt)}")
        if delta_tools:
            print(f"\n  工具调用分布:")
            for tool, counts in sorted(delta_tools.items()):
                for mode, cnt in sorted(counts.items()):
                    if cnt > 0:
                        print(f"    {tool}: {mode}={int(cnt)}")
        print("=" * 60 + "\n")
        return checks

# 运行评测
async def main():
    baseline = get_metrics_snapshot()
    print("📸 已捕获 Prometheus 指标基线\n")

    runtime = PersistentGraphRuntime()
    await runtime.initialize()
    
    # 诊断：验证 Langfuse 配置是否生效
    from app.observability.langfuse_client import get_langfuse_client
    client = get_langfuse_client()
    current_trace = client.get_current_trace_id()
    print(f"🔍 Langfuse 初始化状态: current_trace_id={current_trace}")
    
    evaluator = Evaluator(runtime)
    await evaluator.run_all()
    print(evaluator.generate_report())
    
    evaluator.verify_metrics(baseline)

    await runtime.close()
    print("📤 正在 flush Langfuse 数据到云端...")
    flush_langfuse()

if __name__ == "__main__":
    asyncio.run(main())