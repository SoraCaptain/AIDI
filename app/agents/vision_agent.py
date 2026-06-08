import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.tools.cv_tools import detect_blur
from app.tools.vlm_tools import ask_vlm
from app.memory.short_memory import ShortMemory


load_dotenv()


class VisionAgent:
    def __init__(self):
        self.memory = ShortMemory(max_turns=6)

        self.model = ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0,
        )

        self.agent = create_agent(
            model=self.model,
            tools=[detect_blur, ask_vlm],
            system_prompt=self._build_system_prompt(),
        )

    def _build_system_prompt(self):
        return (
            "你是一个视觉分析助手。\n"
            "你的任务是分析图像质量和内容。\n\n"

            "工具使用规则：\n"
            "1. 如果用户提到模糊、清晰度、质量 → 必须先调用 detect_blur\n"
            "2. 如果用户问图像内容 → 使用 ask_vlm\n"
            "3. 如果 blur_score 很低，你可以直接得出结论\n"
            "4. 如果信息不足，再调用 ask_vlm\n\n"

            "输出要求：\n"
            "- 简洁\n"
            "- 说明是否调用了工具\n"
        )

    def chat(self, user_input: str, image_path: str | None = None):
        if image_path:
            if os.path.isfile(image_path) and os.path.splitext(image_path)[1] in ['.bmp', '.png', '.jpg', '.jpeg']:
                print('valid image path', image_path)
                self.memory.set_image(image_path)
            else:
                return "Not a valid image path"

        current_img = self.memory.get_image()
        if current_img:
            user_input = (
                f"{user_input}\n\n"
                f"当前图片路径是：{current_img}\n"
                f"如果用户说'这张图'、'它'、'刚才那张图'，都指这个图片。"
            )

        self.memory.add("user", user_input)
        print("debug message:", self.memory.get_messages())
        response = self.agent.invoke({
            "messages": self.memory.get_messages()
        })

        output = response["messages"][-1].content
        self.memory.add("assistant", output)

        return output


if __name__ == "__main__":
    agent = VisionAgent()

    print("Vision Agent started.")
    print("输入图片路径和问题。第二轮如果继续问同一张图，image path 可以留空。")
    print("输入 exit 退出。")

    # question = "这张图模糊吗"
    # image_path = "/home/ziyi/gitlocal/AIDI/dog.jpg"
    # answer = agent.chat(question, image_path=image_path)
    # print("\nAssistant:")
    # print(answer)

    while True:
        image_path = input("\nImage path, empty if same as before: ").strip()

        if image_path.lower() == "exit":
            break

        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if image_path == "":
            image_path = None

        answer = agent.chat(question, image_path=image_path)

        print("\nAssistant:")
        print(answer)

# /srv/new_storage/ziyi/Sunny/dataset/test_set/20251121_test_data/NG-2-ID1043901229953974272-R00-C01-72803B6903748506/over2/index000.png
