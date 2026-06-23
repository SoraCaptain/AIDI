# app/skills/builtin/vision_composer.py
"""
内置视觉复合技能：综合分析图像并生成结构化报告
"""
import json
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from app.skills.base import BaseSkill, SkillInput, SkillOutput
from app.tools.native_vision_tools import (
    vlm_understand_image,
)
from app.mcp_clients.vision_mcp_client import build_vision_mcp_client


class ComprehensiveAnalysisInput(SkillInput):
    """综合分析的输入参数"""
    image_path: str = Field(description="图像的本地路径或 HTTP URL")
    question: str = Field(
        default="详细描述这张图片的内容、主要物体、它们的位置关系以及场景整体理解。",
        description="对图像的分析问题"
    )
    detection_conf_threshold: float = Field(default=0.25, description="检测置信度阈值")
    include_segmentation: bool = Field(default=True, description="是否包含分割掩码")


class ComprehensiveAnalysisSkill(BaseSkill):
    """
    综合图像分析技能：
    1. 目标检测 (detect_objects_yolo via MCP)
    2. 实例分割 (segment_with_sam via MCP)
    3. 视觉语言理解 (vlm_understand_image)
    4. 合并结果生成结构化 Markdown 报告
    """

    def __init__(self):
        self._mcp_client = None
        self._detect_objects_yolo: BaseTool | None = None
        self._segment_with_sam: BaseTool | None = None
        self._mcp_loaded = False

    async def _ensure_mcp_tools(self):
        if self._mcp_loaded:
            return
        self._mcp_client = build_vision_mcp_client()
        mcp_tools = await self._mcp_client.get_tools()
        for t in mcp_tools:
            if t.name.endswith("detect_objects_yolo"):
                self._detect_objects_yolo = t
            elif t.name.endswith("segment_with_sam"):
                self._segment_with_sam = t
        self._mcp_loaded = True

    @property
    def name(self) -> str:
        return "comprehensive_image_analysis"

    @property
    def description(self) -> str:
        return (
            "对图像进行深度综合分析。适合需要完整理解图像内容的场景，"
            "例如：详细图像描述、物体统计、场景关系分析。"
            "该技能会同时执行检测、分割和多模态理解，并生成结构化报告。"
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        return ComprehensiveAnalysisInput

    async def execute(self, **kwargs) -> SkillOutput:
        inputs = ComprehensiveAnalysisInput(**kwargs)
        try:
            await self._ensure_mcp_tools()

            detection_task = self._detect_objects_yolo.ainvoke({
                "image_path": inputs.image_path,
                "conf": inputs.detection_conf_threshold,
                "imgsz": 640,
            })

            segmentation_task = None
            if inputs.include_segmentation:
                segmentation_task = self._segment_with_sam.ainvoke({
                    "image_path": inputs.image_path,
                    "max_masks": 20,
                })

            detection_result = await detection_task
            detections = detection_result if isinstance(detection_result, dict) else json.loads(detection_result)

            if detections.get("labels"):
                object_list = ", ".join(set(detections["labels"][:5]))
                enhanced_question = (
                    f"{inputs.question}\n"
                    f"我检测到图片中有这些物体：{object_list}。请结合这些信息，详细分析场景。"
                )
            else:
                enhanced_question = inputs.question

            vlm_answer = await vlm_understand_image.ainvoke({
                "image_path": inputs.image_path,
                "question": enhanced_question,
            })

            segmentation_result = None
            if segmentation_task:
                seg_raw = await segmentation_task
                segmentation_result = seg_raw if isinstance(seg_raw, dict) else json.loads(seg_raw)

            report = {
                "summary": {
                    "total_objects_detected": len(detections.get("labels", [])),
                    "detected_classes": list(set(detections.get("labels", []))),
                },
                "detection_details": detections,
                "segmentation_details": segmentation_result,
                "vlm_analysis": vlm_answer,
                "metadata": {
                    "skill_name": self.name,
                    "image_path": inputs.image_path,
                },
            }

            markdown_report = f"""
                # 📊 综合图像分析报告

                ## 物体统计
                - 检测到 **{len(detections.get('labels', []))}** 个目标
                - 包含类别: {', '.join(set(detections.get('labels', [])))}

                ## 详细检测结果
                ```json
                {json.dumps(detections, ensure_ascii=False, indent=2)}
                ```
                视觉语言理解 (VLM)
                {vlm_answer}

                分割信息
                { '已生成分割掩码' if segmentation_result else '未启用分割' }
                """
            return SkillOutput(
                    success=True,
                    result={
                        "report": markdown_report,
                        "structured_data": report,
                        "vlm_answer": vlm_answer,
                    },
                    metadata={"mode": "comprehensive"},
                )
        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"综合图像分析失败: {str(e)}",
                metadata={"exception": str(e)}
            )