# app/api/gateway.py

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    HTTPException,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.task_store import task_store
from app.api.schemas import (
    CreateTaskResponse,
    TaskStatusResponse,
    HumanReviewRequest,
    HumanReviewResponse,
    ReportResponse,
)
from app.services.graph_runtime import graph_runtime
from app.observability.langfuse_client import flush_langfuse


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "http://localhost:8400",
)

app = FastAPI(
    title="Visual Inspection Agent Gateway",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境要改成你的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(UPLOAD_DIR)), name="files")


@app.on_event("startup")
async def startup():
    await graph_runtime.initialize()


@app.on_event("shutdown")
async def shutdown():
    flush_langfuse()


def save_upload_file(file: UploadFile, task_id: str) -> tuple[str, str]:
    suffix = Path(file.filename).suffix or ".jpg"
    safe_filename = f"{task_id}{suffix}"
    file_path = UPLOAD_DIR / safe_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = f"{PUBLIC_BASE_URL}/files/{safe_filename}"

    return str(file_path), image_url


async def run_graph_background(task_id: str):
    task = task_store.get(task_id)
    if not task:
        return

    task_store.update(task_id, status="running")

    try:
        result = await graph_runtime.run_task(
            task_id=task_id,
            thread_id=task["thread_id"],
            question=task["question"],
            image_url=task["image_url"],
        )

        task_store.update(
            task_id,
            status=result["status"],
            interrupt=result.get("interrupt"),
            result=result.get("raw_result"),
            final_answer=result.get("final_answer"),
            trace_summary=result.get("trace_summary"),
            error=None,
        )

    except Exception as e:
        task_store.update(
            task_id,
            status="failed",
            error=repr(e),
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "visual-inspection-agent-gateway",
    }


@app.post("/tasks", response_model=CreateTaskResponse)
async def create_task(
    background_tasks: BackgroundTasks,
    question: str = Form(...),
    file: UploadFile = File(...),
):
    """
    创建视觉分析任务。

    上传图片 + question，立即返回 task_id。
    后台执行 LangGraph。
    """

    task_id = str(uuid.uuid4())
    thread_id = f"thread-{task_id}"

    image_path, image_url = save_upload_file(file, task_id)

    task_store.create_task(
        task_id=task_id,
        thread_id=thread_id,
        question=question,
        image_path=image_path,
        image_url=image_url,
        session_id=graph_runtime.session_id,
    )

    # FastAPI BackgroundTasks 会在响应发送后执行任务，
    # 适合“先返回 task_id，再后台处理耗时流程”的场景。
    background_tasks.add_task(run_graph_background, task_id)

    task_store.update(task_id, status="queued")

    return CreateTaskResponse(
        task_id=task_id,
        thread_id=thread_id,
        status="queued",
        image_url=image_url,
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    task = task_store.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task["task_id"],
        thread_id=task["thread_id"],
        status=task["status"],
        question=task["question"],
        image_url=task.get("image_url"),
        interrupt=task.get("interrupt"),
        final_answer=task.get("final_answer"),
        error=task.get("error"),
        trace_summary=task.get("trace_summary"),
    )


@app.post("/tasks/{task_id}/human-review", response_model=HumanReviewResponse)
async def submit_human_review(
    task_id: str,
    req: HumanReviewRequest,
):
    """
    提交人工复核输入，恢复 LangGraph。

    action:
    - accept
    - edit
    - retry
    - reject
    """

    task = task_store.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != "waiting_human":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not waiting for human review. Current status: {task['status']}",
        )

    resume_value = {
        "action": req.action,
        "feedback": req.feedback or "",
    }

    if req.action == "edit":
        if not req.edited_answer:
            raise HTTPException(
                status_code=400,
                detail="edited_answer is required when action=edit",
            )

        resume_value["edited_answer"] = req.edited_answer

    task_store.update(task_id, status="running")

    try:
        result = await graph_runtime.resume_task(
            task_id=task_id,
            thread_id=task["thread_id"],
            question=task["question"],
            image_url=task["image_url"],
            resume_value=resume_value,
        )

        task_store.update(
            task_id,
            status=result["status"],
            interrupt=result.get("interrupt"),
            result=result.get("raw_result"),
            final_answer=result.get("final_answer"),
            trace_summary=result.get("trace_summary"),
            error=None,
        )

    except Exception as e:
        task_store.update(
            task_id,
            status="failed",
            error=repr(e),
        )

        raise HTTPException(status_code=500, detail=repr(e))

    return HumanReviewResponse(
        task_id=task_id,
        status=task_store.get(task_id)["status"],
        message="Human review submitted.",
    )


@app.get("/tasks/{task_id}/report", response_model=ReportResponse)
def get_report(task_id: str):
    task = task_store.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return ReportResponse(
        task_id=task_id,
        status=task["status"],
        final_answer=task.get("final_answer"),
    )


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": task_store.list_tasks()
    }