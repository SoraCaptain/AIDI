# app/api/schemas.py

from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class CreateTaskResponse(BaseModel):
    task_id: str
    thread_id: str
    status: str
    image_url: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    thread_id: str
    status: str
    question: str
    image_url: Optional[str] = None
    interrupt: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
    trace_summary: Optional[Dict[str, Any]] = None


class HumanReviewRequest(BaseModel):
    action: str
    feedback: Optional[str] = ""
    edited_answer: Optional[str] = None


class HumanReviewResponse(BaseModel):
    task_id: str
    status: str
    message: str


class ReportResponse(BaseModel):
    task_id: str
    status: str
    final_answer: Optional[str] = None