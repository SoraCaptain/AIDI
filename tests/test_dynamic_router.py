# tests/test_dynamic_router.py
import asyncio
import uuid
from app.services.graph_runtime_persistent import PersistentGraphRuntime

async def test_router():
    runtime = PersistentGraphRuntime()
    await runtime.initialize()

    # 测试 Query 1
    try:
        result = await runtime.run_task(
            task_id="test_router_1",
            thread_id=f"thread_router_{uuid.uuid4().hex[:8]}",
            question="找到这张图里的所有人的准确位置",
            image_url="test_imgs/Family.jpg"
        )
        print(f"结果: {result['final_answer']}")
        print(result)
    except Exception as e:
        print(f"Query 1 错误: {e}")

    # 测试 Query 2
    # try:
    #     result = await runtime.run_task(
    #         task_id="test_router_2",
    #         thread_id=f"thread_router_{uuid.uuid4().hex[:8]}",
    #         question="检测并分割图片中的所有物体",
    #         image_url="test.jpg"
    #     )
    #     print(f"结果: {result['final_answer']}")
    # except Exception as e:
    #     print(f"Query 2 错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_router())