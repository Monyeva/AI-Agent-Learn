"""非交互式测试脚本：验证 Chroma 版 RAG 的建库、检索与持久化。"""
import sys

# 复用 rag_chroma 里已经写好的函数（导入时会自动创建 client / collection / 加载 embedding 模型）
from rag_chroma import (
    KNOWLEDGE_FILE,
    load_documents,
    chunk_text,
    build_index,
    retrieve,
    generate_answer,
    collection,
)

query = sys.argv[1] if len(sys.argv) > 1 else "这家公司靠什么赚钱？"

print("=" * 60)
print(f"当前向量库数据条数：{collection.count()}")

if collection.count() == 0:
    print("向量库为空 -> 开始切分 + 向量化 + 写入 Chroma ...")
    text = load_documents(KNOWLEDGE_FILE)
    chunks = chunk_text(text)
    print(f"切分出 {len(chunks)} 个文本块")
    build_index(chunks)
else:
    print("向量库已有数据 -> 直接从磁盘加载，跳过重新 embedding")

print("=" * 60)
print(f"问题：{query}")
retrieved = retrieve(query, top_k=2)
print("\n检索到的最相关资料：")
for dist, c in retrieved:
    print(f"  [距离 {dist:.4f}，相似度 {1 - dist:.3f}] {c[:60]}...")

print("\n回答：")
print(generate_answer(query, retrieved))
