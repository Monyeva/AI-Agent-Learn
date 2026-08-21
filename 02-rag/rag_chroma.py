"""
第三个项目：RAG + 向量数据库（Chroma）

相比 minimal_rag.py（向量存在「内存列表」里），向量数据库解决：
1. 持久化：数据存磁盘，程序重启还在，不用每次重新 embedding
2. 高效检索：用 ANN 索引（近似最近邻），数据量大也不怕线性扫描慢
3. 动态增删：随时 add / delete / update
4. 元数据：每条向量可附带元数据（来源、时间等），支持过滤

这里用 Chroma（最简单、Python 友好、可本地持久化）。
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# ============ 配置 ============
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "<默认 key>"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1"),
)
MODEL = os.getenv("MODEL", "deepseek-v4-pro")
KNOWLEDGE_FILE = "data/knowledge.txt"

# ============ embedding 模型 ============
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def load_documents(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """按段落切分，再把小段落合并到接近 chunk_size（避免切断语义边界）。

    相比纯固定长度切分的优势：像「定价」这种短段落，不会再被硬塞进
    一个和「技术支持 / 融资」混在一起的大块里，导致语义被稀释。
    只有单个段落本身超过 chunk_size 时，才对它退化为固定长度 + 重叠硬切。
    """
    # 1. 按空行拆成段落（知识库里段落之间是空行）
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 2. 贪心合并小段落，直到接近 chunk_size
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

    # 3. 对仍然超长的单个段落，退化为固定长度 + 重叠硬切
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
    """文档块 -> 向量列表。"""
    vectors = embed_model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list:
    """问题 -> 向量。"""
    return embed_model.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True).tolist()


# ============ 向量数据库（Chroma） ============
# 持久化到本地 ./chroma_db 目录，程序重启后数据还在
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 集合（类似数据库的「表」），用余弦距离衡量相似度
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},
)


def build_index(chunks: list):
    """把 chunks 向量化后写入向量库。每条 = id + 文档 + 向量。"""
    vectors = embed_documents(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=vectors,
    )
    print(f"已写入 {len(chunks)} 条向量到 Chroma")


def retrieve(query: str, top_k: int = 2) -> list:
    """把问题向量化，从 Chroma 检索 top_k 个最相似的文档。"""
    q_vec = embed_query(query)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
    )
    docs = results["documents"][0]
    distances = results["distances"][0]  # 余弦距离：越小越相似
    return list(zip(distances, docs))


def generate_answer(query: str, retrieved: list) -> str:
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


def main():
    text = load_documents(KNOWLEDGE_FILE)
    chunks = chunk_text(text)
    print(f"共 {len(chunks)} 个文本块")

    # 关键：向量库已有数据就跳过 embedding（演示持久化的价值）
    if collection.count() == 0:
        print("向量库为空，开始向量化并写入...")
        build_index(chunks)
    else:
        print(f"向量库已有 {collection.count()} 条数据，直接从磁盘加载（无需重新 embedding）")

    while True:
        query = input("\n你的问题（输入 exit 退出）：\n> ")
        if query.lower() == "exit":
            break

        retrieved = retrieve(query, top_k=2)
        print("\n检索到的最相关资料：")
        for dist, c in retrieved:
            print(f"  [相似度 {1 - dist:.3f}] {c[:50]}...")

        answer = generate_answer(query, retrieved)
        print("\n回答：")
        print(answer)


if __name__ == "__main__":
    main()
