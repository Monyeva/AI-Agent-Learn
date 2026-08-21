"""
第四个内容：用 LangGraph 重构 Agent

把之前手写的 while 循环，改写成一张「图」：
    节点（做什么）+ 边（下一步去哪）

核心对照：
    手写版 while 循环        -> 图里的环 agent -> tools -> agent
    手写版 messages 列表     -> State（状态，用 add_messages 累加）
    手写版 调用模型          -> 节点 call_model
    手写版 if tool_calls     -> 条件边 should_continue（路由）
    手写版 for 执行工具      -> 工具节点 ToolNode
"""

import os
from typing import Annotated, TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ============ 1. 配置 ============
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "<默认 key>"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1"),
    model=os.getenv("MODEL", "deepseek-v4-pro"),
)

SYSTEM_PROMPT = "你是一个有用的助手。当用户需要计算、查询时间或统计文字数量时，请调用对应工具。"


# ============ 2. 定义工具（@tool 装饰器自动生成 name / description） ============
@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，例如 "3 * (4 + 5)"。"""
    # 注意：这里用 eval 只是为了演示最简效果。
    # 生产环境绝对不能直接 eval 用户输入，会有严重安全风险。
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算出错: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def word_count(text: str) -> str:
    """计算文本中的中文字数量。"""
    return "中文字数量是：" + str(len(text))


TOOLS = [calculator, get_current_time, word_count]

# 把工具绑定到模型：等价于手写版调用时传 tools=TOOLS
llm_with_tools = llm.bind_tools(TOOLS)


# ============ 3. 定义 State（Agent 的短期记忆） ============
class AgentState(TypedDict):
    # add_messages 是「累加器」：节点返回的新消息不是覆盖，而是追加到历史里
    # 这就是手写版 messages.append(...) 在框架里的实现
    messages: Annotated[list, add_messages]


# ============ 4. 定义节点 ============
def call_model(state: AgentState):
    """节点1：调用模型。对应手写版 chat.completions.create(...)。"""
    # 返回 {"messages": [...]}，会被 add_messages 累加进 state
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 工具节点：LangGraph 内置，自动执行 message.tool_calls 里的工具
# 对应手写版 for tool_call in message.tool_calls: result = func(**args)
tool_node = ToolNode(TOOLS)


# ============ 5. 定义路由（条件边） ============
def should_continue(state: AgentState):
    """对应手写版 if message.tool_calls 判断。"""
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"   # 还要调工具 -> 去工具节点
    return "end"         # 没有工具调用 -> 结束


# ============ 6. 组装图 ============
graph = StateGraph(AgentState)

# 加节点
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

# 加边
graph.add_edge(START, "agent")   # 从 START 进入 agent
graph.add_conditional_edges(     # agent 之后根据结果路由
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
graph.add_edge("tools", "agent")  # 工具执行完回到 agent（这就是循环）

# 编译成可运行的 agent
agent = graph.compile()


def run_agent(user_input: str) -> str:
    """把用户问题交给图，返回最终回答。"""
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]
        }
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    # 改法1：多轮对话 —— 自己维护 messages 历史，跨轮保留（实现「记忆」）
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    print("多轮对话 Agent（输入 exit 退出）")
    while True:
        question = input("\n你：\n> ")
        if question.lower() == "exit":
            break
        # 关键：把本轮问题「追加」进历史，而不是每轮都新建 messages
        messages.append(HumanMessage(content=question))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]   # 把完整历史（含本轮回答）存回 messages
        print("\nAgent：")
        print(result["messages"][-1].content)
