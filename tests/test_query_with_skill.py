# tests/test_query_with_skill.py
import asyncio
from app.services.graph_runtime_persistent import PersistentGraphRuntime
from utils.logger import logger

async def run_query(question: str, image_url: str):
    runtime = PersistentGraphRuntime()
    await runtime.initialize()

    # 准备一个唯一的 task_id 和 thread_id
    task_id = "test-skill-001"
    thread_id = "thread-skill-001"

    try:
        result = await runtime.run_task(
            task_id=task_id,
            thread_id=thread_id,
            question=question,
            image_url=image_url,
        )

        logger.info("\n" + "=" * 50)
        logger.info("📊 执行结果：")
        logger.info("=" * 50)
        logger.info(f"状态: {result.get('status')}")
        if result.get('final_answer'):
            logger.info(f"最终答案:\n{result.get('final_answer')}")
        else:
            logger.info(f"完整结果: {result}")
    finally:
        await runtime.close()

if __name__ == "__main__":
    # 运行查询 1
    logger.info("\n🚀 执行查询 1：全面报告\n")
    asyncio.run(run_query(
        "请对这张图片 test_imgs/Family.jpg 生成一份全面的分析报告，我需要知道里面有哪些人、他们的表情状态、画面中的文字内容以及整体场景氛围。",
        "test_imgs/Family.jpg"
    ))

    # 运行查询 2（可以注释掉第一个，单独测试第二个）
    # print("\n🚀 执行查询 2：简洁分析\n")
    # asyncio.run(run_query(
    #     "看图说话，分析一下 test_photo.png 的主要内容和可读文字。",
    #     "test_photo.png"
    # ))