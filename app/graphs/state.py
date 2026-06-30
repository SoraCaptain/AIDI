from typing import TypedDict, Optional, List, Dict, Any


class VisionGraphState(TypedDict, total=False):
    # request
    session_id: str
    question: str
    image_path: Optional[str]

    # memory
    conversation_history: List[Dict[str, Any]]
    last_result: Optional[str]
    memory_context: Dict[str, Any]
    memory_stats: Dict[str, Any]

    # planner
    plan: Dict[str, Any]
    required_agents: List[str]
    planner_reason: str

    # agent outputs
    quality_result: Optional[str]
    ocr_result: Optional[str]
    detection_result: Optional[str]
    segmentation_result: Optional[str]
    vlm_result: Optional[str]
    memory_result: Optional[str]

    # agent errors
    quality_error: Optional[str]
    ocr_error: Optional[str]
    detection_error: Optional[str]
    segmentation_error: Optional[str]
    vlm_error: Optional[str]
    memory_error: Optional[str]

    # aggregate
    aggregated_result: Optional[str]

    # critic / HITL
    critic_decision: Optional[str]
    critic_reason: Optional[str]
    human_decision: Optional[str]
    human_feedback: Optional[str]
    human_edited_answer: Optional[str]

    # control
    retry_count: int
    max_retries: int

    # dynamic router
    execution_plan: Optional[Any]
    agent_results: Optional[Dict[str, Any]]

    # final
    final_answer: Optional[str]
    task_id: Optional[str]