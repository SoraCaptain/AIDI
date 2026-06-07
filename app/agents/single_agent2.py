import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


def build_agent():
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=[add, multiply],
        system_prompt=(
            "你是一个严谨的工程助手。"
            "当问题需要计算时，你必须优先调用工具，而不是自己心算。"
            "回答要简洁，并说明你用了哪个工具。"
        ),
    )

    return agent


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

    print(result["messages"][-1].content)