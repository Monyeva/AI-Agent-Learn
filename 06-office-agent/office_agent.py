"""
办公自动化 Agent：用自然语言指挥工具完成办公任务。

能力：
- 读 Excel（read_excel）
- 写 Excel 报表（write_excel，自动保存到 output/）
- 查 SQLite 业务库（query_database，仅允许 SELECT）
- 发邮件（send_email，未配置真实 SMTP 时模拟保存到 outbox/）
- 列出可用文件（list_office_files）

Agent 框架：LangGraph（StateGraph + ToolNode + 条件边）。
命令行体验入口：cli.py
"""

import os
import sqlite3
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Annotated, TypedDict

import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# ============ 路径配置 ============

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTBOX_DIR = os.path.join(BASE_DIR, "outbox")
DB_PATH = os.path.join(DATA_DIR, "company.db")

# ============ LLM 配置 ============

api_key = os.getenv("OPENAI_API_KEY", "<默认 key>")
base_url = os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1")
model = os.getenv("MODEL", "deepseek-v4-pro")


def _resolve_excel(path_or_name: str) -> str:
    """把文件名或路径解析成实际存在的 Excel 路径。"""
    if os.path.isfile(path_or_name):
        return path_or_name
    candidate = os.path.join(DATA_DIR, path_or_name)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"找不到文件：{path_or_name}（可先调用 list_office_files 查看可用文件）")


# ============ 工具 ============

@tool
def list_office_files() -> str:
    """列出办公目录下可用的文件（data 和 output）。当你不知道有哪些文件可以操作时调用。"""
    lines = []
    for d in (DATA_DIR, OUTPUT_DIR):
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name)
            size = os.path.getsize(path)
            lines.append(f"{path}（{size} 字节）")
    return "\n".join(lines) if lines else "（目前没有可用文件）"


@tool
def read_excel(file_path: str, max_rows: int = 30) -> str:
    """读取 Excel 文件并返回文本表格。file_path 可以是文件名（如 sales_data.xlsx）或完整路径。"""
    path = _resolve_excel(file_path)
    df = pd.read_excel(path)
    return df.head(max_rows).to_string(index=False)


@tool
def write_excel(file_name: str, columns: list, rows: list) -> str:
    """把表格数据写入 Excel，自动保存到 output 目录。
    file_name 是文件名（无需路径，如 sales_summary.xlsx，可省略 .xlsx 后缀）；
    columns 是列名列表；rows 是二维列表，每行一个列表，元素顺序必须与 columns 一致。"""
    if not file_name.lower().endswith(".xlsx"):
        file_name += ".xlsx"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, file_name)
    df = pd.DataFrame(rows, columns=columns)
    df.to_excel(path, index=False)
    return f"✅ 已写入报表：{os.path.abspath(path)}（{len(rows)} 行 × {len(columns)} 列）"


@tool
def aggregate_excel(file_name: str, group_by: str = "", value_column: str = "", agg: str = "sum") -> str:
    """对 Excel 做分组聚合，返回结果表（用 pandas 精确计算，不要心算）。
    file_name 是 Excel 文件名；group_by 是分组列名（留空则对整个表聚合一行）；
    value_column 是数值列名；agg 是聚合方式：sum / mean / count / max / min。
    示例：aggregate_excel('sales_data.xlsx', '地区', '销售额', 'sum')"""
    path = _resolve_excel(file_name)
    df = pd.read_excel(path)
    if agg not in ("sum", "mean", "count", "max", "min"):
        return f"不支持的聚合方式：{agg}（可选 sum/mean/count/max/min）"
    if value_column and value_column not in df.columns:
        return f"列不存在：{value_column}。可用列：{list(df.columns)}"
    if group_by and group_by not in df.columns:
        return f"分组列不存在：{group_by}。可用列：{list(df.columns)}"
    if group_by:
        result = df.groupby(group_by)[value_column].agg(agg).reset_index()
    else:
        result = pd.DataFrame({f"{agg}({value_column})": [getattr(df[value_column], agg)()]})
    return result.to_string(index=False)


@tool
def query_database(sql: str) -> str:
    """对业务数据库执行 SELECT 查询，返回结果表格。
    只允许 SELECT；可用表：employees(id, name, dept, position, salary) 员工表、
    products(id, name, category, price) 产品表。"""
    sql_clean = sql.strip()
    if not sql_clean.lower().startswith("select"):
        return "错误：只允许 SELECT 查询。"
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(sql_clean, conn)
        if df.empty:
            return "（查询结果为空）"
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL 执行失败：{e}"
    finally:
        conn.close()


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件。
    若设置了 SMTP_HOST 环境变量则真实发送；否则模拟发送——把邮件保存到 outbox 目录并如实告知。
    to 是收件人邮箱地址，subject 是主题，body 是正文。"""
    smtp_host = os.getenv("SMTP_HOST", "")
    if smtp_host:
        sender = os.getenv("SMTP_USER", "")
        password = os.getenv("SMTP_PASSWORD", "")
        port = int(os.getenv("SMTP_PORT", "465"))
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        with smtplib.SMTP_SSL(smtp_host, port) as server:
            server.login(sender, password)
            server.sendmail(sender, [to], msg.as_string())
        return f"✅ 邮件已通过 SMTP 发送给 {to}（主题：{subject}）"

    # 模拟发送：保存到 outbox/
    os.makedirs(OUTBOX_DIR, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{to}.txt"
    path = os.path.join(OUTBOX_DIR, filename)
    content = f"收件人: {to}\n主题: {subject}\n发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n正文:\n{body}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"📨 邮件已模拟发送并保存到 {os.path.abspath(path)}（未配置 SMTP_HOST，未真实发送）"


TOOLS = [list_office_files, read_excel, aggregate_excel, write_excel, query_database, send_email]

# ============ System Prompt ============

SYSTEM_PROMPT = """你是「启明星科技」的办公自动化助手，用自然语言指挥工具完成办公任务。回答时用中文，步骤之间简短说明你正在做什么。

可用的数据（先用 list_office_files 查看有哪些文件）：
- data/sales_data.xlsx：销售流水，列 = 日期, 地区, 产品, 销售员, 数量, 单价, 销售额
  - 地区：华东 / 华北 / 华南 / 西南；产品：星图平台 / 数据分析服务 / 培训服务 / 定制开发
- data/company.db：SQLite 业务库
  - employees(id, name, dept, position, salary, email)：员工，部门有 技术部 / 销售部 / 市场部 / 财务部，email 是员工邮箱
  - products(id, name, category, price)：产品目录

工作规则：
1. 不确定文件结构时，先 read_excel 或 query_database 查看数据，不要凭空猜测。
2. 任何汇总/统计/求和/平均，一律用 aggregate_excel（Excel）或 SQL 聚合（数据库），由工具精确计算，**绝不自己心算**。拿到工具返回的数字后，原样引用。
3. 需要生成报表时，用 write_excel 保存到 output/（file_name 只给文件名，不要带路径）。
4. 数据库查询只能用 SELECT（query_database 会强制校验）。
5. 发邮件用 send_email；未配置真实 SMTP 时会模拟保存到 outbox，要如实告诉用户这是模拟。
6. 每个任务完成后，告诉用户结果和生成的文件路径。"""

# ============ Agent 图 ============

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0.2)
llm_with_tools = llm.bind_tools(TOOLS)


def call_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "end"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


agent = build_agent()


def chat(message: str, history: list = None) -> str:
    """处理一条用户消息（history 为之前对话的 [{"role","content"}] 列表），返回回答文本。"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in (history or []):
        role = item.get("role")
        if role == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif role == "assistant":
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=message))
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("办公自动化 Agent 快速演示（python office_agent.py）\n")
    for q in [
        "华东地区的总销售额是多少？",
        "技术部的平均薪资是多少？",
        "把每个地区的销售额汇总成 Excel 报表",
        "给张伟发一封邮件，提醒他本周提交市场周报",
    ]:
        print(f"\n用户：{q}")
        print("-" * 60)
        print(chat(q))
