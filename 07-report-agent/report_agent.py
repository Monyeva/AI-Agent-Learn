"""
多 Agent 调研报告生成器 —— 手写 Supervisor（监督者）模式

架构：
    用户任务
       │
       ▼
    Supervisor（主管 Agent）—— 决定派给谁
       │  researcher_tool      writer_tool      finish_tool
       ▼                       ▼                ▼
    研究员 Agent             报告撰写 Agent      完成
    (search_kb + SQL)       (write_report)
       └──────────► Supervisor ◄──────────┘

核心思想（Supervisor 模式）：
- 每个 worker 是一个独立的 ReAct Agent，各带自己的工具（分工明确）
- Supervisor 也是一个 LLM，但它不带业务工具，而是把「每个 worker」声明成自己的工具
- Supervisor 通过 tool_calls 决定「下一步派哪个 worker」，worker 执行完回来，Supervisor 再决策
- 直到 Supervisor 觉得信息够了，调用 finish_tool 结束

和 LangGraph 官方 supervisor 教程一致：worker 作为 supervisor 的「工具」，
supervisor 通过 bind_tools 感知 worker 的职责，用 tool_calls 编排多 Agent。
"""

import os
import sqlite3
import json
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from typing import Annotated, TypedDict

# ============ 1. 路径 & LLM 配置 ============

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
KB_PATH = os.path.join(DATA_DIR, "company_kb.txt")
DB_PATH = os.path.join(DATA_DIR, "company.db")

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "<默认 key>"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1"),
    model=os.getenv("MODEL", "deepseek-v4-pro"),
    temperature=0.2,
)


# ============ 2. 研究员 Agent 的工具（检索知识库 + 查数据库） ============

def _load_kb() -> str:
    with open(KB_PATH, encoding="utf-8") as f:
        return f.read()


# 中文查询切词：把常见连接词/功能词当分隔符，切出名词片段（教学演示够用）
_SPLIT_WORDS = "的、和、与、及、或、是、在、要、请、了、帮我、请帮我、一下、关于、跟、对、给、把、为、做、写、调研、报告".replace("、", " ")


def _tokenize(query: str) -> list:
    """把中文查询切成分词片段。例：「启明星科技的产品和定价」→ [启明星科技, 产品, 定价]"""
    import re
    q = re.sub(r"[，。？、！；：,.?!;:\s]+", " ", query)
    for w in sorted(_SPLIT_WORDS.split(), key=len, reverse=True):
        q = q.replace(w, " ")
    return [w.strip() for w in q.split() if w.strip()]


@tool
def search_company_kb(query: str) -> str:
    """在启明星科技的公司资料知识库中检索与 query 相关的段落。
    资料涵盖：公司简介、星图平台产品、定价策略、融资历程、核心团队、市场表现、竞争优势、客户案例、行业趋势。
    当需要了解公司产品、定价、融资、团队、市场、竞争等信息时调用。"""
    kb = _load_kb()
    paragraphs = [p.strip() for p in kb.split("\n\n") if p.strip()]
    words = _tokenize(query)
    if not words:
        words = [query]
    scored = []
    for para in paragraphs:
        score = 0
        title = para.split("\n")[0]
        for w in words:
            if w in title:
                score += 2
            if w in para:
                score += 1
        scored.append((score, para))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [p for s, p in scored if s > 0][:3]
    if not hits:
        # 兜底：查询中任意 2~4 字连续片段出现在段落标题里就算命中
        # 例：查询「公司团队情况」→ 片段「团队」出现在标题「核心团队」→ 命中
        fallback = []
        for p in paragraphs:
            title = p.split("\n")[0]
            matched = any(
                query[i:j] in title
                for i in range(len(query) - 1)
                for j in range(i + 2, min(len(query) + 1, i + 5))
            )
            if matched:
                fallback.append(p)
            if len(fallback) == 2:
                break
        if not fallback:
            return f"知识库中没有检索到与「{query}」相关的段落。"
        return "\n\n".join(f"【资料{i+1}】\n{p}" for i, p in enumerate(fallback))
    return "\n\n".join(f"【资料{i+1}】\n{p}" for i, p in enumerate(hits))


@tool
def query_sales_data(region: str = "", product: str = "", agg: str = "sum") -> str:
    """查询销售数据库并返回聚合结果。
    region 可选：华东/华北/华南/西南（留空则所有地区）；product 可选：星图平台/定制开发/培训服务（留空则所有产品）；
    agg 可选：sum（销售额总量）/ count（订单数）/ mean（平均销售额）。"""
    if agg not in ("sum", "count", "mean"):
        return f"不支持的聚合方式：{agg}（可选 sum/count/mean）"
    sql = "SELECT "
    sql += agg.upper() + "(amount)" if agg in ("sum", "mean") else "COUNT(*)"
    sql += " FROM sales"
    conds, params = [], []
    if region:
        conds.append("region = ?")
        params.append(region)
    if product:
        conds.append("product = ?")
        params.append(product)
    if conds:
        sql += " WHERE " + " AND ".join(conds)

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(sql, params)
        val = cur.fetchone()[0]
        agg_name = {"sum": "总销售额", "count": "订单数", "mean": "平均销售额"}[agg]
        label = f"{region or '所有地区'} / {product or '所有产品'}"
        return f"{label}：{agg_name} = {val:,.0f} 元（若为 count 则单位是笔）"
    except Exception as e:
        return f"SQL 执行失败：{e}"
    finally:
        conn.close()


@tool
def query_top_products() -> str:
    """返回销售额 TOP 产品排行（按全年总销售额降序）。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT product, SUM(amount) FROM sales GROUP BY product ORDER BY SUM(amount) DESC"
        )
        rows = cur.fetchall()
        return "\n".join(f"{p}: {a:,.0f} 元" for p, a in rows)
    except Exception as e:
        return f"SQL 执行失败：{e}"
    finally:
        conn.close()


RESEARCHER_TOOLS = [search_company_kb, query_sales_data, query_top_products]


# ============ 3. 报告撰写 Agent 的工具 ============

@tool
def write_report(file_name: str, markdown_content: str) -> str:
    """把调研结果写成 Markdown 报告并保存到 output 目录。
    file_name 是文件名（无需路径，如 market_report.md，可省略 .md 后缀）；
    markdown_content 是完整的 Markdown 报告正文。"""
    if not file_name.lower().endswith(".md"):
        file_name += ".md"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, file_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    return f"✅ 报告已保存：{os.path.abspath(path)}"


WRITER_TOOLS = [write_report]


# ============ 4. 构建两个 worker Agent ============

# 每个 worker 是一个独立的 ReAct Agent（create_react_agent），自带工具、自带循环
researcher_agent = create_react_agent(llm, tools=RESEARCHER_TOOLS)
writer_agent = create_react_agent(llm, tools=WRITER_TOOLS)


# ============ 5. Supervisor：把 worker 包装成「工具」 ============

# 关键：Supervisor 不带业务工具，它把每个 worker 当成一个工具来「调用」。
# 这样 Supervisor 只负责「编排」，具体干活的是 worker。

@tool
def researcher(query: str) -> str:
    """研究员 Agent：负责收集信息。当需要从公司知识库或销售数据库获取资料时，把调研问题交给它。
    它会自行调用 search_company_kb / query_sales_data / query_top_products 等工具完成调研，返回调研结论。"""
    result = researcher_agent.invoke(
        {"messages": [("human", f"请调研以下问题，结合知识库和销售数据给出结论：{query}")]}
    )
    return result["messages"][-1].content


@tool
def writer(report_request: str) -> str:
    """报告撰写 Agent：负责把调研结论写成正式 Markdown 报告。
    给它一个「报告需求」（包含标题、要涵盖的要点、已有的调研数据），它会把报告写到 output/ 并返回文件路径。
    写报告前请先确保调研 Agent 已经收集到足够信息。"""
    result = writer_agent.invoke(
        {"messages": [("human", f"请写一份正式调研报告：{report_request}")]}
    )
    return result["messages"][-1].content


@tool
def finish(summary: str) -> str:
    """所有工作已完成，调用此工具结束任务。传入最终报告路径或总结。"""
    return f"✅ 任务完成：{summary}"


SUPERVISOR_TOOLS = [researcher, writer, finish]


# ============ 6. Supervisor 提示词 ============

SUPERVISOR_SYSTEM = """你是一个多 Agent 团队的「主管 Supervisor」，负责把用户任务拆解并派发给两个 worker。

可用 worker：
1. researcher：研究员，收集信息（公司资料、销售数据）。接到调研任务后自动检索知识库和数据库。
2. writer：报告撰写员，把调研结论整理成正式的 Markdown 报告并保存到 output/。
3. finish：调用它代表任务完成，传入最终报告路径或总结。

工作流程：
- 用户给一个调研报告主题后，先派 researcher 收集必要信息（可分多次、多个角度调研）。
- 信息足够后，派 writer 写报告（把调研到的数据、要点、标题都传给 writer）。
- writer 返回报告路径后，调用 finish 结束。

要求：
- 不要代替 worker 干活，你的职责是「决定派给谁、传什么任务」，不是自己回答。
- 一次只调用一个工具，等结果返回后再决定下一步。
- 报告要求内容详实、数据准确，引用真实检索到的数据，不编造。
"""


# ============ 7. LangGraph 图：Supervisor 编排多个 worker ============

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def supervisor_node(state: AgentState):
    """Supervisor 节点：把完整对话（含各 worker 的结果）交给 supervisor LLM，让它决策下一步。"""
    result = supervisor_llm.invoke(state["messages"])
    return {"messages": [result]}


def should_continue(state: AgentState):
    """条件边：Supervisor 有工具调用就去 worker；其中含 finish 则结束。"""
    last = state["messages"][-1]
    if last.tool_calls:
        names = [tc["name"] for tc in last.tool_calls]
        if "finish" in names:
            return "finish"
        return "worker"   # researcher / writer 都是「执行一个 worker」
    return "end"


def worker_node(state: AgentState):
    """worker 节点：逐个执行 Supervisor 点名要调用的 worker 工具，把结果作为 ToolMessage 喂回。"""
    last = state["messages"][-1]
    funcs = {t.name: t for t in SUPERVISOR_TOOLS}
    tool_messages = []
    for tool_call in last.tool_calls:
        # 跳过 finish（finish 由 finish 节点处理）
        if tool_call["name"] == "finish":
            continue
        func = funcs.get(tool_call["name"])
        if func is None:
            result = f"错误：未知工具 {tool_call['name']}"
        else:
            result = func.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": tool_messages}


# Supervisor LLM 绑定 worker 工具（worker 对 supervisor 而言只是工具）
supervisor_llm = llm.bind_tools(SUPERVISOR_TOOLS)

tool_node = ToolNode([researcher, writer, finish])

graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("worker", worker_node)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    should_continue,
    {"worker": "worker", "finish": "finish", "end": END},
)
# worker 执行完回到 supervisor（这就是编排循环）
graph.add_edge("worker", "supervisor")

# finish 节点：记录最终输出后结束
def finish_node(state: AgentState):
    last = state["messages"][-1]
    # 找到 finish 那个 tool_call 并回应
    finish_calls = [tc for tc in last.tool_calls if tc["name"] == "finish"]
    if finish_calls:
        args = finish_calls[0].get("args", {})
        summary = args.get("summary", "任务完成") if isinstance(args, dict) else "任务完成"
    else:
        summary = "任务完成"
    # 补全所有 tool_call 的响应，避免 OpenAI 校验失败
    tool_msgs = [
        ToolMessage(content=str(summary), tool_call_id=tc["id"])
        for tc in last.tool_calls
    ]
    return {"messages": tool_msgs}

graph.add_node("finish", finish_node)
graph.add_edge("finish", END)

agent = graph.compile()


# ============ 8. 对话入口 ============

def run_report_task(task: str) -> str:
    """输入一个调研报告主题，运行多 Agent 协作流程，返回最终报告路径/总结。"""
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        {"role": "user", "content": task},
    ]
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


def chat(message: str, history: list = None) -> str:
    """多轮对话入口。history 为 [{"role","content"}] 列表。"""
    messages = [{"role": "system", "content": SUPERVISOR_SYSTEM}]
    for item in (history or []):
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": message})
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("多 Agent 调研报告生成器 —— Supervisor 模式 快速演示\n")
    task = "调研启明星科技的市场表现，写一份 2024 年市场调研报告"
    print(f"任务：{task}\n" + "-" * 60)
    print(run_report_task(task))
