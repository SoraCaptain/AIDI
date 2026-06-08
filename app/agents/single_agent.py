import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# Qwen3 / vLLM XML tool‑call parser
# ---------------------------------------------------------------------------
# When vLLM serves Qwen3 models with --tool-call-parser qwen3_coder and
# tool_choice="auto" (the LangChain default), the model returns tool calls
# as XML inside the `content` field while `tool_calls` stays empty.
# LangChain therefore never executes the tools.
#
# This middleware detects that situation, extracts the JSON tool‑call
# payloads from the XML, and populates `tool_calls` so the agent works
# correctly without forcing tool_choice="required" (which would prevent
# the model from giving a final text answer after tool results).
# ---------------------------------------------------------------------------

_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def _parse_qwen_xml_tool_calls(content: str) -> list[dict[str, Any]]:
    """Extract OpenAI‑style tool‑call dicts from Qwen XML content."""
    tool_calls: list[dict[str, Any]] = []
    for i, raw in enumerate(_TOOL_CALL_XML_RE.findall(content)):
        try:
            data = json.loads(raw.strip())
            tool_calls.append(
                {
                    "name": data["name"],
                    "args": data.get("arguments", {}),
                    "id": f"qwen_call_{i}_{data['name']}",
                    "type": "tool_call",
                }
            )
        except (json.JSONDecodeError, KeyError):
            pass  # ignore unparseable blocks
    return tool_calls


class QwenToolCallParsingMiddleware(AgentMiddleware):
    """Middleware that converts Qwen3 XML tool calls in content → tool_calls."""

    def wrap_model_call(
        self, request: ModelRequest, handler: Any
    ) -> ModelResponse:
        response: ModelResponse = handler(request)

        # Work on a copy of the result list so we can replace messages
        new_messages: list[BaseMessage] = []
        for msg in response.result:
            if isinstance(msg, AIMessage) and msg.content:
                # Only intervene when the model returned text but no
                # structured tool_calls were recognised by the LLM provider.
                if not msg.tool_calls and "<tool_call>" in str(msg.content):
                    parsed = _parse_qwen_xml_tool_calls(str(msg.content))
                    if parsed:
                        # Replace the message with one that carries tool_calls
                        # and clears the text content (tool calls are not
                        # meant to be displayed).
                        new_msg = AIMessage(
                            content="",
                            tool_calls=parsed,
                            id=msg.id,
                        )
                        new_messages.append(new_msg)
                        continue
            new_messages.append(msg)

        return ModelResponse(
            result=new_messages,
            structured_response=response.structured_response,
        )


class ToolUsageReminderMiddleware(AgentMiddleware):
    """Middleware that appends tool-usage info when the model forgets to mention it.

    After the agent finishes (no more tool_calls), this middleware checks the
    conversation history for preceding ToolMessage entries.  If any tools were
    used but the final AIMessage does not contain the name of at least one of
    them, the middleware appends a brief "使用的工具：..." note.
    """

    def wrap_model_call(
        self, request: ModelRequest, handler: Any
    ) -> ModelResponse:
        response: ModelResponse = handler(request)

        new_messages: list[BaseMessage] = []
        for msg in response.result:
            if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                # This is a final text answer (no further tool calls).
                # Look back through the full message history to find which
                # tools were called earlier in this turn.
                used_tools: list[str] = []
                for m in request.messages:
                    tool_name = getattr(m, "name", None)
                    if tool_name and tool_name not in used_tools:
                        used_tools.append(tool_name)

                if used_tools:
                    content = str(msg.content)
                    # Only append if the model didn't already mention the tool
                    already_mentioned = any(
                        tool_name in content for tool_name in used_tools
                    )
                    if not already_mentioned:
                        tool_list = "、".join(used_tools)
                        new_content = f"使用的工具：{tool_list}\n{content}"
                        new_msg = AIMessage(
                            content=new_content,
                            id=msg.id,
                        )
                        new_messages.append(new_msg)
                        continue

            new_messages.append(msg)

        return ModelResponse(
            result=new_messages,
            structured_response=response.structured_response,
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------


def build_agent():
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=[add, multiply],
        middleware=[
            QwenToolCallParsingMiddleware(),
            ToolUsageReminderMiddleware(),  # 如果用qwen3.6-30B 可以不需要这个
        ],
        system_prompt=(
            "你是一个严谨的工程助手。"
            "当问题需要计算时，你必须优先调用工具，而不是自己心算。"
            "【重要】每次回答必须严格遵循以下格式，先说明工具使用，再给答案：\n"
            "使用的工具：<工具名>\n"
            "答案：<你的答案>"
        ),
    )

    return agent


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = build_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请计算 23 * 17，再加上 42。",
                }
            ]
        }
    )

    print("Final answer:", result["messages"][-1].content)