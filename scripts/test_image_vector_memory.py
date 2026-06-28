from app.memory.image_vector_memory import ImageVectorMemory
from utils.logger import logger

ivm = ImageVectorMemory(db_path="data/memory/vision_memory.sqlite3")

results = ivm.search_similar_images(
    image_path="/home/ziyi/gitlocal/AIDI/test_imgs/WDLD14439B1A_16-Cam1-1226-3.bmp",
    top_k=5,
    min_score=0.1,
)

for item in results:
    logger.info("=" * 80)
    logger.info(f"score: {item["image_similarity_score"]}")
    logger.info(f"image_path: {item["image_path"]}")
    logger.info(f"task_id: {item["task_id"]}")
    logger.info(f"question: {item["question"]}")
    logger.info(f"final_answer: {item["final_answer"][:300] if item["final_answer"] else None}")