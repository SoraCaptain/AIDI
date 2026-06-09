# app/graphs/vision_graph.py
#   START
#     ↓
#  prepare
#     ↓
# blur_check
#     ↓
# route_after_blur
#     ├─ blur_bad → report
#     ├─ need_vlm → vlm_analyze
#     └─ no_image → report
#               ↓
#         validate_result
#               ↓
#         route_after_validate
#           ├─ pass → report
#           ├─ retry → vlm_analyze
#           └─ fail → report
#               ↓
#             END

from typing import TypedDict, Optional, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from app.tools.cv_api import detect_blur_api
from app.tools.vlm_api import ask_vlm_api


load_dotenv()


class VisionState(TypedDict, total=False):
    # 用户输入
    question: str
    image_path: Optional[str]

    # 短期上下文
    history: List[dict]

    # CV 结果
    blur_score: Optional[float]
    is_blurry: Optional[bool]

    # VLM 结果
    vlm_answer: Optional[str]

    # 控制字段
    route: Optional[str]
    retry_count: int
    max_retries: int
    error: Optional[str]

    # 输出
    final_answer: Optional[str]


def prepare_node(state: VisionState) -> dict:
    """
    准备节点：
    - 初始化 retry_count
    - 初始化 history
    - 检查是否有图片
    """

    updates = {}

    if "retry_count" not in state:
        updates["retry_count"] = 0

    if "max_retries" not in state:
        updates["max_retries"] = 2

    if "history" not in state:
        updates["history"] = []

    question = state.get("question", "")
    image_path = state.get("image_path")

    if not image_path:
        updates["route"] = "no_image"
    else:
        updates["route"] = "has_image"

    updates["history"] = state.get("history", []) + [
        {
            "node": "prepare",
            "question": question,
            "image_path": image_path,
        }
    ]

    return updates


def blur_check_node(state: VisionState) -> dict:
    """
    模糊检测节点：
    调用 OpenCV blur API。
    """

    image_path = state.get("image_path")

    if not image_path:
        return {
            "error": "No image_path provided.",
            "route": "no_image",
        }

    try:
        result = detect_blur_api(image_path)

        blur_score = float(result.get("blur_score", 0.0))
        is_blurry = bool(result.get("is_blurry", False))

        return {
            "blur_score": blur_score,
            "is_blurry": is_blurry,
            "history": state.get("history", []) + [
                {
                    "node": "blur_check",
                    "blur_score": blur_score,
                    "is_blurry": is_blurry,
                }
            ],
        }

    except Exception as e:
        return {
            "error": f"blur_check failed: {repr(e)}",
            "history": state.get("history", []) + [
                {
                    "node": "blur_check",
                    "error": repr(e),
                }
            ],
        }


def route_after_blur(state: VisionState) -> str:
    """
    第一个条件分支：
    - 没图片：直接报告
    - blur tool 失败：尝试走 VLM，看是否还能分析
    - 图片明显模糊：直接报告
    - 否则：调用 VLM 做语义分析
    """

    if state.get("route") == "no_image":
        return "report"

    if state.get("error"):
        return "vlm_analyze"

    if state.get("is_blurry") is True:
        return "report"

    return "vlm_analyze"


def vlm_analyze_node(state: VisionState) -> dict:
    """
    VLM 分析节点：
    调用 VLM Server。
    """

    image_path = state.get("image_path")
    question = state.get("question", "请分析这张图片是否有明显问题。")

    retry_count = state.get("retry_count", 0)

    if not image_path:
        return {
            "error": "No image_path provided for VLM.",
        }

    prompt = (
        "你是一个视觉质检助手。\n"
        "请基于图片回答用户问题，并特别关注：\n"
        "1. 图像内容\n"
        "2. 是否有明显质量问题\n"
        "3. 是否存在缺陷、异常、遮挡、模糊、曝光问题\n\n"
        f"用户问题：{question}\n"
    )

    try:
        answer = ask_vlm_api(image_path=image_path, question=prompt)

        return {
            "vlm_answer": answer,
            "history": state.get("history", []) + [
                {
                    "node": "vlm_analyze",
                    "retry_count": retry_count,
                    "vlm_answer_preview": answer[:200],
                }
            ],
        }

    except Exception as e:
        return {
            "error": f"vlm_analyze failed: {repr(e)}",
            "retry_count": retry_count + 1,
            "history": state.get("history", []) + [
                {
                    "node": "vlm_analyze",
                    "retry_count": retry_count,
                    "error": repr(e),
                }
            ],
        }


def validate_result_node(state: VisionState) -> dict:
    """
    校验节点：
    判断 VLM 结果是否足够可用。
    这里先用简单规则，后面会升级成 Critic Agent。
    """

    answer = state.get("vlm_answer")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not answer:
        if retry_count < max_retries:
            return {
                "route": "retry",
                "retry_count": retry_count + 1,
                "history": state.get("history", []) + [
                    {
                        "node": "validate_result",
                        "decision": "retry_no_answer",
                        "retry_count": retry_count + 1,
                    }
                ],
            }
        else:
            return {
                "route": "fail",
                "history": state.get("history", []) + [
                    {
                        "node": "validate_result",
                        "decision": "fail_no_answer",
                    }
                ],
            }

    if len(answer.strip()) < 20:
        if retry_count < max_retries:
            return {
                "route": "retry",
                "retry_count": retry_count + 1,
                "history": state.get("history", []) + [
                    {
                        "node": "validate_result",
                        "decision": "retry_too_short",
                        "retry_count": retry_count + 1,
                    }
                ],
            }
        else:
            return {
                "route": "fail",
                "history": state.get("history", []) + [
                    {
                        "node": "validate_result",
                        "decision": "fail_too_short",
                    }
                ],
            }

    return {
        "route": "pass",
        "history": state.get("history", []) + [
            {
                "node": "validate_result",
                "decision": "pass",
            }
        ],
    }


def route_after_validate(state: VisionState) -> str:
    """
    第二个条件分支：
    - pass：生成报告
    - retry：回到 VLM 节点
    - fail：生成失败报告
    """

    route = state.get("route")

    if route == "retry":
        return "vlm_analyze"

    return "report"


def report_node(state: VisionState) -> dict:
    """
    最终报告节点。
    这一版先用规则生成报告。
    后面会换成 Report Agent。
    """

    question = state.get("question", "")
    image_path = state.get("image_path")
    blur_score = state.get("blur_score")
    is_blurry = state.get("is_blurry")
    vlm_answer = state.get("vlm_answer")
    error = state.get("error")
    route = state.get("route")

    lines = []

    lines.append("## 视觉分析报告")
    lines.append("")
    lines.append(f"- 图片路径：{image_path}")
    lines.append(f"- 用户问题：{question}")
    lines.append("")

    if blur_score is not None:
        lines.append("### 图像清晰度")
        lines.append(f"- blur_score: {blur_score:.2f}")
        lines.append(f"- 是否模糊: {is_blurry}")
        lines.append("")

    if is_blurry is True:
        lines.append("### 初步结论")
        lines.append("图片疑似模糊，建议重新采集或提高图像质量后再做进一步视觉分析。")
        lines.append("")

    if vlm_answer:
        lines.append("### VLM 分析")
        lines.append(vlm_answer)
        lines.append("")

    if error:
        lines.append("### 错误信息")
        lines.append(error)
        lines.append("")

    if route == "fail":
        lines.append("### 状态")
        lines.append("分析结果未通过校验，且已达到最大重试次数。")
        lines.append("")

    lines.append("### 执行轨迹")
    for item in state.get("history", []):
        lines.append(f"- {item}")

    return {
        "final_answer": "\n".join(lines)
    }


def build_vision_graph():
    graph = StateGraph(VisionState)

    graph.add_node("prepare", prepare_node)
    graph.add_node("blur_check", blur_check_node)
    graph.add_node("vlm_analyze", vlm_analyze_node)
    graph.add_node("validate_result", validate_result_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "blur_check")

    # blur_check 完成后，不是固定去某个节点
    # 而是调用 route_after_blur(state)
    # 根据返回值决定下一步

    graph.add_conditional_edges(
        "blur_check",
        route_after_blur,
        {
            "report": "report",
            "vlm_analyze": "vlm_analyze",
        },
    )

    graph.add_edge("vlm_analyze", "validate_result")

    graph.add_conditional_edges(
        "validate_result",
        route_after_validate,
        {
            "vlm_analyze": "vlm_analyze",
            "report": "report",
        },
    )

    graph.add_edge("report", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_vision_graph()

    print("Vision Graph v1 started.")
    print("输入 exit 退出。")

    current_image_path = None

    while True:
        image_path = input("\nImage path, empty if same as before: ").strip()

        if image_path.lower() == "exit":
            break

        if image_path:
            current_image_path = image_path

        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        result = app.invoke(
            {
                "image_path": current_image_path,
                "question": question,
                "retry_count": 0,
                "max_retries": 2,
                "history": [],
            }
        )

        print("\n" + "=" * 80)
        print(result["final_answer"])
        print("=" * 80)
        
# /home/ziyi/gitlocal/AIDI/test_imgs/train01.png