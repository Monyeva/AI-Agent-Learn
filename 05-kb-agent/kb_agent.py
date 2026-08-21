"""
企业知识库问答 Agent —— 核心逻辑

能力：
1. RAG：加载 / 切分 / embedding / 向量库 / 检索（带来源标注）
2. Agent：LangGraph 工具调用（search_knowledge_base 等）
3. 引用溯源：回答标注【资料N】，并可提取检索到的原始资料
4. 文档上传：动态把新文档加入知识库（用 source 元数据记录来源）
"""

import os
import time
from typing import Annotated, TypedDict
from datetime import datetime

# 国内镜像 + 离线加载（必须在加载模型前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "data", "knowledge.txt")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

SYSTEM_PROMPT = (
    "你是一个企业知识库问答助手，可以调用工具回答问题。\n"
    "可用工具：\n"
    "1. search_knowledge_base：检索知识库（默认企业资料 + 用户上传的文档）。当问题涉及公司产品、定价、融资、团队、业务等时调用。\n"
    "2. calculator：计算数学表达式。\n"
    "3. get_current_time：获取当前时间。\n"
    "回答要求：\n"
    "- 优先基于检索到的资料回答；引用资料时标注来源，例如：【资料1】\n"
    "- 回答知识库问题时，请尽量只调用一次 search_knowledge_base，调用前想好一个最完整、最可能命中资料的问题（top_k 会一次返回多条）\n"
    "- 如果检索结果不足以回答，请如实说明，不要编造。"
)


# ============ 2. RAG：embedding + 向量库 ============
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},
)


def load_documents(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """按段落切分：小段落合并、超长段落才硬切，避免切断语义边界。"""
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


def build_index(chunks: list, source: str = "默认知识库"):
    """把文本块向量化并写入向量库，附带来源元数据。"""
    vectors = embed_documents(chunks)
    ids = [f"{source}_{int(time.time())}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]
    collection.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
    print(f"已写入 {len(chunks)} 条向量（来源：{source}）")


def retrieve(query: str, top_k: int = 3) -> list:
    """检索 top_k 个最相关的文本块，返回 (距离, 文本, 元数据)。"""
    q_vec = embed_query(query)
    results = collection.query(query_embeddings=[q_vec], n_results=top_k)
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]
    return list(zip(dists, docs, metas))


def ensure_index():
    """确保默认知识库已建立索引（幂等）。"""
    if collection.count() == 0:
        text = load_documents(KNOWLEDGE_FILE)
        chunks = chunk_text(text)
        build_index(chunks, source="默认知识库")
        print(f"默认知识库索引完成：{len(chunks)} 块")
    else:
        print(f"向量库已有 {collection.count()} 条数据，跳过建索引")


def add_document(file_path: str, source_name: str = None) -> int:
    """把一个文档加入知识库，返回加入的块数。"""
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return 0
    chunks = chunk_text(text)
    source = source_name or os.path.basename(file_path)
    vectors = embed_documents(chunks)
    ids = [f"{source}_{int(time.time())}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]
    collection.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
    return len(chunks)


def list_sources() -> dict:
    """返回知识库里的文档来源统计：{来源名: 块数}。"""
    data = collection.get(include=["metadatas"])
    counts = {}
    for meta in data["metadatas"]:
        src = (meta or {}).get("source", "未知来源")
        counts[src] = counts.get(src, 0) + 1
    return counts


# ============ 3. 工具 ============
@tool
def search_knowledge_base(query: str) -> str:
    """检索知识库（默认企业资料 + 用户上传的文档），返回相关片段。当问题与知识库内容相关时调用。"""
    retrieved = retrieve(query, top_k=5)
    if not retrieved:
        return "知识库中没有检索到相关信息。"
    parts = []
    for i, (dist, doc, meta) in enumerate(retrieved):
        source = (meta or {}).get("source", "未知来源")
        parts.append(f"【资料{i + 1}】来源《{source}》\n{doc}")
    return "\n\n".join(parts)


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


TOOLS = [search_knowledge_base, calculator, get_current_time]


# ============ 4. LangGraph Agent ============
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


# ============ 5. 对话 + 引用溯源 ============
def extract_sources(result) -> list:
    """从图执行结果里提取所有检索工具返回的资料片段（用于引用溯源）。"""
    sources = []
    seen = set()
    for m in result["messages"]:
        if isinstance(m, ToolMessage) and m.content:
            key = m.content[:80]
            if key not in seen:
                seen.add(key)
                sources.append(m.content)
    return sources


def chat(message: str, history: list) -> tuple:
    """多轮对话入口。

    history 可以是 [(user, assistant), ...] 或 [{"role":..., "content":...}, ...]。
    返回 (回答, 资料来源 Markdown)。
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in history:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
        else:
            role, content = item
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))
    result = agent.invoke({"messages": messages})

    answer = result["messages"][-1].content
    sources = extract_sources(result)
    if sources:
        source_md = "**📎 本次回答参考的资料：**\n\n" + "\n\n---\n\n".join(f"> {s}" for s in sources)
    else:
        source_md = "（本次回答未检索知识库）"
    return answer, source_md


if __name__ == "__main__":
    ensure_index()
    print("\n来源统计：", list_sources())
    ans, src = chat("启明星科技的主打产品是什么？", [])
    print("\n回答：\n", ans)
    print("\n" + src)
