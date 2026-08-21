"""
第五个内容：RAG + Agent 结合（知识库问答 Agent）

把前面两个东西拼成一个完整项目：
    1. RAG（检索增强生成）：从向量库检索知识
    2. Agent（工具调用）：模型决定要不要调工具、调哪个

核心变化：把「检索」包装成一个工具 search_knowledge_base，
让 Agent 自己判断「用户问的是不是知识库问题」，是就去检索，不是就不检索。

这样 Agent 能同时：
    - 问「启明星科技的主打产品」  -> 检索知识库回答
    - 问「计算 3*5」              -> 调 calculator
    - 问「现在几点」              -> 调 get_current_time

注意：图和循环逻辑（LangGraph 部分）和 03 完全一样，
只是 TOOLS 列表里多了一个检索工具 —— 这就是框架「加能力 = 加工具」的价值。
"""

import os

# 国内镜像 + 离线加载（和 02-rag 一样，必须在加载模型前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from typing import Annotated, TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from sentence_transformers import SentenceTransformer
import chromadb

# ============ 1. 配置 ============
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "<默认 key>"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1"),
    model=os.getenv("MODEL", "deepseek-v4-pro"),
)
KNOWLEDGE_FILE = "data/knowledge.txt"

SYSTEM_PROMPT = (
    "你是一个有用的助手，可以调用工具来完成任务。\n"
    "可用工具：\n"
    "1. search_knowledge_base：检索「启明星科技」的公司资料（产品、定价、融资、团队等）\n"
    "2. calculator：计算数学表达式\n"
    "3. get_current_time：获取当前时间\n"
    "4. word_count：统计文字数量\n"
    "当用户问到启明星科技相关的问题时，请先调用 search_knowledge_base 检索，再根据检索结果回答。"
)


# ============ 2. RAG：embedding + 向量库 ============
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},
)


def load_documents(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """按段落切分（同 02-rag 的优化版），小段落合并、超长段才硬切。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 1 <= chunk_size:
            current = current + "\n" + para
        else:
            merged.append(current)
            current = para
    if current:
        merged.append(current)

    chunks = []
    for block in merged:
        if len(block) <= chunk_size:
            chunks.append(block)
        else:
            start = 0
            block_len = len(block)
            while start < block_len:
                end = min(start + chunk_size, block_len)
                piece = block[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= block_len:
                    break
                start = end - overlap
    return chunks


def embed_documents(texts: list) -> list:
    vectors = embed_model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list:
    return embed_model.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True).tolist()


def build_index(chunks: list):
    vectors = embed_documents(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=vectors)
    print(f"已写入 {len(chunks)} 条向量到 Chroma")


def retrieve(query: str, top_k: int = 2) -> list:
    q_vec = embed_query(query)
    results = collection.query(query_embeddings=[q_vec], n_results=top_k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(distances, docs))


# ============ 3. 工具：把「检索」包装成工具 ============
@tool
def search_knowledge_base(query: str) -> str:
    """检索「启明星科技」的公司资料（产品、定价、融资、团队等），返回相关资料。当用户问启明星科技相关问题时调用。"""
    retrieved = retrieve(query, top_k=2)
    if not retrieved:
        return "知识库中没有检索到相关信息。"
    return "\n\n".join(f"【资料{i+1}】\n{doc}" for i, (_, doc) in enumerate(retrieved))


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，例如 "3 * (4 + 5)"。"""
    # 注意：eval 仅用于演示，生产环境不能直接 eval 用户输入
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


TOOLS = [search_knowledge_base, calculator, get_current_time, word_count]


# ============ 4. LangGraph Agent（和 03 一样，只是多了检索工具） ============
llm_with_tools = llm.bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(TOOLS)


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "end"


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
agent = graph.compile()


# ============ 5. 主流程 ============
def main():
    # 建索引（已有就跳过）
    if collection.count() == 0:
        text = load_documents(KNOWLEDGE_FILE)
        chunks = chunk_text(text)
        print(f"共切分 {len(chunks)} 个文本块，开始建索引...")
        build_index(chunks)
    else:
        print(f"向量库已有 {collection.count()} 条数据，直接从磁盘加载")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    print("\n知识库问答 Agent（输入 exit 退出）")
    while True:
        question = input("\n你：\n> ")
        if question.lower() == "exit":
            break
        messages.append(HumanMessage(content=question))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print("\nAgent：")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
