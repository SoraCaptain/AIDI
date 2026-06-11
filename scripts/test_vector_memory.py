from app.memory.vector_memory import VectorMemory

vm = VectorMemory(db_path="data/memory/vision_memory.sqlite3")

results = vm.search_similar_tasks(
    query="这张图是否有划痕或表面缺陷？",
    top_k=5,
    min_score=0.2,
)

for item in results:
    print("=" * 80)
    print("score:", item["similarity_score"])
    print("task_id:", item["task_id"])
    print("question:", item["question"])
    print("final_answer:", item["final_answer"][:300] if item["final_answer"] else None)
