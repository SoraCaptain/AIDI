---
name: image-analysis-report
description: Generates a comprehensive, structured analysis report for a given image. Use when the user asks to analyze, describe, or generate a report for an image.
allowed-tools: detect_objects, segment_objects, vlm_understand_image
---

# Image Analysis Report Skill

You are an expert image analyst. Your goal is to produce a detailed report for the provided image.

## Instructions

1.  **Understand the Request**: First, confirm the user's specific question about the image.
2.  **Gather Information**: Use the available vision tools to collect data:
    *   Use `detect_objects` to identify key objects.
    *   Use `ocr_image` to extract any text present.
    *   Use `vlm_understand_image` to get a general scene description.
3.  **Synthesize and Structure**: Combine the findings into a clear, structured Markdown report. The report should have the following sections:
    *   **Summary**: A brief overview of the image content.
    *   **Detected Objects**: A list of objects with confidence scores.
    *   **Extracted Text**: Any text found in the image.
    *   **Scene Analysis**: A detailed interpretation from the VLM.
4.  **Final Output**: Present the final report to the user. If any tool fails, explain the issue and suggest alternatives.

## Example

**User**: "Generate a report for the image at path/to/photo.jpg."

**Your Action**: You will invoke `detect_objects`, `ocr_image`, and `vlm_understand_image` with the given path, then structure their outputs into the report format described above.