import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# --- 1. 配置LLM客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址
API_KEY = os.getenv("API_KEY")
# 初始化ModelScope客户端
client = OpenAI(
    base_url="https://api-inference.modelscope.cn/v1",
    api_key=API_KEY,
)


def chat_with_llm():
    # 初始化消息历史
    messages = []

    print("=== 欢迎使用 DeepSeek 聊天机器人 ===")
    print("输入 'exit' 退出程序")
    print("输入 'clear' 清空对话历史")
    print("输入 'save' 保存对话记录")
    print("-" * 50)

    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ")

            # 检查特殊命令
            if user_input.lower() == "exit":
                print("再见！")
                break
            elif user_input.lower() == "clear":
                messages = []
                print("对话历史已清空")
                continue
            elif user_input.lower() == "save":
                with open("chat_history.txt", "w", encoding="utf-8") as f:
                    for msg in messages:
                        f.write(f"{msg['role']}: {msg['content']}\n")
                print("对话已保存到 chat_history.txt")
                continue
            elif user_input.strip() == "":
                continue

            # 添加用户消息到历史
            messages.append({"role": "user", "content": user_input})

            # 调用API获取回复
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True,
                temperature=0.7,  # 控制创造性
                max_tokens=2000,  # 限制生成长度
            )

            # 处理流式响应
            assistant_response = ""
            print("助手: ", end="", flush=True)

            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    assistant_response += content
                    print(content, end="", flush=True)

            # 添加助手回复到历史
            messages.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\n\n中断操作，输入'exit'退出程序")
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            # 移除最后一条用户消息（因为回复失败）
            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    chat_with_llm()
