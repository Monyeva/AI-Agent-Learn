# 企业知识库问答 Agent

一个完整的企业知识库问答 Agent：**RAG 检索 + LangGraph 工具调用 + 引用溯源 + 量化评测 + Gradio Web 界面**。

内置「启明星科技」企业资料作为示例知识库，支持上传自有文档，开箱即用。

## 功能亮点

| 亮点 | 说明 |
|---|---|
| 🧠 **Agentic RAG** | 检索封装为 `search_knowledge_base` 工具，Agent 自行判断何时检索、检索什么 |
| 📎 **引用溯源** | 回答自动标注【资料N】，并在界面下方展示参考的原文片段 |
| 📊 **量化评测** | 检索命中率 + 回答正确率 + 关键词召回率，结果写入 `eval/report.json` |
| 💬 **多轮记忆** | LangGraph 消息累加，支持多轮追问（携带上文） |
| 📤 **文档上传** | 上传 `.txt/.md` 自动切块 + 向量化入知识库，保留来源名 |

## 目录结构

```
05-kb-agent/
├── app.py                  # Gradio Web 界面（3 个 Tab）
├── kb_agent.py             # 核心：RAG + LangGraph Agent + 引用溯源 + 文档上传
├── evaluate.py             # 量化评测脚本
├── requirements.txt
├── data/
│   └── knowledge.txt       # 示例企业知识库
├── eval/
│   ├── qa_pairs.json       # 评测集（10 条 QA 对）
│   └── report.json         # 评测报告（自动生成）
├── chroma_db/              # 向量库（自动生成）
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

首次运行需下载 embedding 模型 `BAAI/bge-small-zh-v1.5`（已在本机缓存，离线可加载）。

### 2. 配置 LLM

默认在 `kb_agent.py` 中通过环境变量配置（已带默认值）：

```python
api_key = os.getenv("OPENAI_API_KEY", "<默认 key>")
base_url = os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1")
model    = os.getenv("MODEL", "deepseek-v4-pro")
```

如需覆盖，运行前设置环境变量即可，无需改代码。

### 3. 启动 Web 界面

```bash
python app.py
```

浏览器打开 http://127.0.0.1:7860

- **智能问答**：多轮对话，回答自动标注【资料N】并展示引用原文
- **知识库管理**：上传 `.txt/.md` 文档加入知识库，查看各来源块数
- **量化评测**：展示评测报告，可一键重跑

### 4. 命令行评测

```bash
python evaluate.py
```

逐条打印命中情况，汇总指标写入 `eval/report.json`。

### 5. 命令行快速体验

```bash
python kb_agent.py
```

建索引 + 一次示例问答，打印引用溯源。

## 评测结果（内置 10 条 QA 对）

| 指标 | 含义 | 结果 |
|---|---|---|
| 检索命中率 | 期望关键词是否全部出现在 top_k=5 检索结果里 | **90%** |
| 回答正确率 | Agent 回答是否包含所有期望关键词 | **100%** |
| 关键词召回率 | 全部关键词的命中比例 | **90%** |

> 唯一未命中（技术负责人 Q9）：200 字切块把「向量检索」从「向」字处切断，关键词跨块边界，
> 但 Agent 回答仍能正确给出。属切块边界问题，见下方「优化方向」。

## 架构

```
用户问题
   │
   ▼
┌──────────── LangGraph Agent ────────────┐
│  LLM(bind_tools) ── 判断是否调用工具      │
│   ┌───────────┐  tool_calls  ┌────────┐  │
│   │   agent   │ ───────────► │  tools │  │
│   └─────┬─────┘              └───┬────┘  │
│         │◄────────────────────────┘      │
└─────────┴───────────────────────────────┘
   │ 检索结果（含来源）
   ▼
┌──────────── RAG ────────────┐
│  BGE embedding（bge-small-zh）│
│  Chroma 向量库（cosine）      │
│  段落感知切块（200 字/50 重叠）│
└─────────────────────────────┘
```

- **切块**：段落感知——短段落合并、超长段落才硬切，避免切断语义边界
- **检索**：BGE 中文 embedding + Chroma 余弦相似度，top_k=5
- **Agent**：LangGraph `StateGraph`，`should_continue` 条件边决定继续调工具还是结束
- **引用溯源**：从图执行结果的 `ToolMessage` 中提取检索原文

## 已知局限 & 优化方向

1. **切块边界切断词**：200 字硬切可能把词组（如「向量检索」）从中间切开 → 增大 overlap 或按句切分
2. **回答非确定性**：LLM 生成每次略不同，同一题的「回答正确率」可能波动 → 多轮取平均
3. **评测为关键词级**：期望关键词需精确出现（已做空格/字符变体归一化）→ 可升级为语义级评测（如 RAGAS）
4. **无 Reranker**：检索结果按 embedding 相似度排序，未做精排 → 可加 BGE-Reranker 提升相关性
5. **单轮记忆在内存**：会话记忆存于 `gr.State`，服务重启即清空 → 可换 LangGraph Checkpointer + 持久化

## 免责声明

- 知识库与 QA 集均为教学演示用虚构数据
- `calculator` 工具使用 `eval`，仅限可信输入场景，生产环境需替换为安全求值
- LLM key 为本地网关默认值，请按需替换
