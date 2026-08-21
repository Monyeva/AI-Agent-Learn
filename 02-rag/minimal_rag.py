"""
第二个项目：最小化 RAG（检索增强生成）

RAG 解决 LLM 的三大问题：知识截止、幻觉、不知道私有数据。
核心思路：回答前先从知识库「检索」相关内容，再让模型「照着资料」回答。

五步流程：
    1. 加载文档 Load
    2. 切分文本 Chunk
    3. 向量化 Embed
    4. 存储 Store
    5. 检索+生成 Retrieve + Generate

向量化已经升级为「真 embedding」（本地 BGE 模型 bge-small-zh-v1.5）。
它能把语义相近的文字映射到相近的向量（比如「赚钱」和「定价」），
比之前的词袋模型（只能匹配字符是否相同）强得多。
"""

import os
import math

# 国内镜像：必须在加载模型前设置，否则默认从 HuggingFace 官网下载（国内可能慢/失败）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型已下载后，用离线模式加载，跳过联网校验（否则网络波动会导致卡住超时）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ============ 配置 ============
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "<默认 key>"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1"),
)
MODEL = os.getenv("MODEL", "deepseek-v4-pro")

KNOWLEDGE_FILE = "data/knowledge.txt"   # 你的知识库文件


# ============ 第 1 步：加载文档 ============
def load_documents(path: str) -> str:
    """读入知识库文件，返回全部文字。"""
    with open(path, encoding="utf-8") as f:
        return f.read()


# ============ 第 2 步：切分文本 ============
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """按固定长度把文档切块，块与块之间有重叠。

    为什么固定长度：段落长短不一，太长的段落会混合多个主题，向量被稀释。
    为什么重叠：固定长度会从中间切断句子，重叠能让边界附近的语义不丢失。

    两个超参数：
      - chunk_size：每块最多多少字（太小会碎片化，太大会稀释）
      - overlap：相邻两块重叠多少字（太小会丢边界，太大会冗余）
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


# ============ 第 3 步：向量化（真 embedding） ============
# BGE 中文 embedding 模型：把文字变成 512 维语义向量（只加载一次）
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# BGE 模型的要求：给「问题」加这个前缀，文档不用加，检索效果会更好
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def embed_documents(texts: list) -> list:
    """把一批文档块变成向量列表。文档不加前缀。"""
    vectors = embed_model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list:
    """把问题变成一个向量。问题要加前缀。"""
    vec = embed_model.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)
    return vec.tolist()


# ============ 第 4 步：相似度检索 ============
def cosine_similarity(a: list, b: list) -> float:
    """余弦相似度：衡量两个向量方向有多接近。范围 [-1, 1]，越接近 1 越相似。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(query: str, chunks: list, vectors: list, top_k: int = 2) -> list:
    """把问题也向量化，找出最相似的 top_k 个 chunk。"""
    q_vec = embed_query(query)
    scored = []
    for i, v in enumerate(vectors):
        score = cosine_similarity(q_vec, v)
        scored.append((score, chunks[i]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ============ 第 5 步：生成（检索 + LLM 回答） ============
def generate_answer(query: str, retrieved: list) -> str:
    """把检索到的资料拼进 prompt，让模型照着资料回答。"""
    context = "\n\n".join(f"【资料{i+1}】\n{c}" for i, (_, c) in enumerate(retrieved))
    prompt = f"""请只根据下面提供的资料回答问题。如果资料里没有答案，请如实说「资料中没有相关信息」，不要编造。

{context}

问题：{query}
回答："""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ============ 主流程 ============
def main():
    text = load_documents(KNOWLEDGE_FILE)
    chunks = chunk_text(text)
    print(f"共加载 {len(chunks)} 个文本块")
    for i, c in enumerate(chunks[:]):
        print(f"  [块 {i+1}] {c[:]}...")

    vectors = embed_documents(chunks)
    print(f"向量化完成，向量维度：{len(vectors[0])}")

    while True:
        query = input("\n你的问题（输入 exit 退出）：\n> ")
        if query.lower() == "exit":
            break

        retrieved = retrieve(query, chunks, vectors, top_k=2)
        print("\n检索到的最相关资料：")
        for score, c in retrieved:
            print(f"  [相似度 {score:.3f}] {c[:60]}...")

        answer = generate_answer(query, retrieved)
        print("\n回答：")
        print(answer)


if __name__ == "__main__":
    main()
