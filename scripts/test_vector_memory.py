from app.memory.vector_memory import VectorMemory
from utils.logger import logger

vm = VectorMemory(db_path="data/memory/vision_memory.sqlite3")

results = vm.search_similar_tasks(
    query="这张图是否有划痕或表面缺陷？",
    top_k=5,
    min_score=0.2,
)

for item in results:
    logger.info("=" * 80)
    logger.info(f"score: {item["similarity_score"]}")
    logger.info(f"task_id: {item["task_id"]}")
    logger.info(f"question: {item["question"]}")
    logger.info(f"final_answer: {item["final_answer"][:300] if item["final_answer"] else None}")
