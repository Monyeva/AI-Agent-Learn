"""
第一个最小化 AI Agent（手写，不依赖任何框架）

这个文件要让你理解 Agent 的核心本质，就一句话：

    Agent = 大模型 + 工具 + 一个 while 循环

循环过程（也叫 ReAct 循环）：
    1. 把用户问题 + 对话历史发给 LLM
    2. LLM 判断：是直接回答，还是需要调用某个工具？
    3. 如果需要工具 -> 我们执行工具 -> 把结果喂回 LLM -> 回到第 2 步
    4. 如果不需要工具 -> 说明 LLM 已经有最终答案 -> 输出

看懂这个文件后，你会发现 LangChain/LangGraph 等框架，
本质上只是把这个循环做得更健壮、更方便而已。
"""

import os
import json
from datetime import datetime

from openai import OpenAI

# ============ 1. 配置：换成你自己的 API ============
# 支持 OpenAI / DeepSeek / 通义 / Kimi / 智谱 等所有「OpenAI 兼容」接口
# 改这里的三样东西即可：api_key、base_url、模型名
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "<默认 key>"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1"),
)
MODEL = os.getenv("MODEL", "deepseek-v4-pro")


# ============ 2. 定义工具（Agent 真正能执行的函数） ============
def calculator(expression: str) -> str:
    """计算一个数学表达式，例如 "3 * (4 + 5)"。"""
    # 注意：这里用 eval 只是为了演示最简效果。
    # 生产环境绝对不能直接 eval 用户输入，会有严重安全风险。
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算出错: {e}"


def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def word_count(text: str) -> str:
    print(f"[调试] word_count 被调用，输入是: {text}")
    """计算文本中的中文字数量。"""
    count = len(text)
    return "中文字数量是：" + str(count)

# 工具的「说明书」：告诉 LLM 有哪些工具、每个工具的参数长什么样。
# LLM 就是靠这段 JSON 来决定「要不要调工具、调哪个、传什么参数」。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 '3*(4+5)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "计算文本中的中文字数量",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要计算中文字数量的文本",
                    }
                },
                "required": ["text"],
            },
        },
    }
]

# 把「工具名字符串」映射到「真正的 Python 函数」
AVAILABLE_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "word_count": word_count,
}

SYSTEM_PROMPT = "你是一个有用的助手。当用户需要计算、查询时间或统计文字数量时，请调用对应工具。"


def run_agent(user_input: str) -> str:
    """Agent 的主循环。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # 最多循环 5 轮，防止模型一直要求调工具导致死循环
    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message

        # 如果模型没要求调用工具，说明它已经给出了最终答案
        if not message.tool_calls:
            return message.content or "（模型没有返回内容）"

        # 把模型这一轮的回复（含 tool_calls）记进对话历史
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [tc.model_dump() for tc in message.tool_calls],
            }
        )

        # 逐个执行模型要求调用的工具
        for tool_call in message.tool_calls:
            func = AVAILABLE_FUNCTIONS[tool_call.function.name]
            args = json.loads(tool_call.function.arguments)
            result = func(**args)

            # 关键一步：把工具的执行结果，以 tool 消息的形式喂回给 LLM
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return "达到最大循环轮数，仍未得到最终答案"


if __name__ == "__main__":
    question = input("你想问什么？\n> ")
    print("\nAgent 回答：")
    print(run_agent(question))
