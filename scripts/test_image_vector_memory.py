from app.memory.image_vector_memory import ImageVectorMemory

ivm = ImageVectorMemory(db_path="data/memory/vision_memory.sqlite3")

results = ivm.search_similar_images(
    image_path="/home/ziyi/gitlocal/AIDI/test_imgs/WDLD14439B1A_16-Cam1-1226-3.bmp",
    top_k=5,
    min_score=0.1,
)

for item in results:
    print("=" * 80)
    print("score:", item["image_similarity_score"])
    print("image_path:", item["image_path"])
    print("task_id:", item["task_id"])
    print("question:", item["question"])
    print("final_answer:", item["final_answer"][:300] if item["final_answer"] else None)