"""
企业知识库问答 Agent —— Gradio Web 界面

三个 Tab：
1. 智能问答：多轮对话（带多轮记忆）+ 引用资料面板
2. 知识库管理：上传文档加入知识库 + 来源统计
3. 量化评测：展示评测报告 + 一键重跑

启动：python app.py
"""

import json
import os

# 必须先于任何第三方库设置：强制离线加载 embedding 模型（模型已缓存本机）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr

import kb_agent as kb
import evaluate

TOP_INFO = (
    "> **企业知识库问答 Agent** —— RAG 检索 + 工具调用 Agent + 引用溯源 + 量化评测。\n"
    "> 已内置「启明星科技」企业资料，也可在「知识库管理」上传自己的文档。"
)

# ============ 工具函数 ============

def format_summary(report: dict) -> str:
    return (
        f"- **总题数**：{report['total_questions']}\n"
        f"- **检索命中率**：{report['retrieval_hit_rate']:.1%}\n"
        f"- **回答正确率**：{report['answer_accuracy']:.1%}\n"
        f"- **关键词召回率**：{report['keyword_recall']:.1%}\n"
        f"- **检索 top_k**：{report['top_k']}\n"
        f"- **生成时间**：{report['generated_at']}"
    )


def format_rows(report: dict) -> list:
    headers = ["问题", "期望关键词", "检索命中", "检索关键词", "回答正确", "回答关键词"]
    rows = []
    for r in report["rows"]:
        rows.append([
            r["question"],
            "、".join(r["expected_keywords"]),
            "✔" if r["retrieval_hit"] else "✘",
            r["retrieval_keywords"],
            "✔" if r["answer_correct"] else "✘",
            r["answer_keywords"],
        ])
    return rows


def load_report() -> tuple:
    """读取最近一次评测报告，返回 (汇总 Markdown, 明细表格)。"""
    if not os.path.exists(evaluate.REPORT_FILE):
        return "（还没有评测报告，请点击下方「运行评测」按钮。）", []
    with open(evaluate.REPORT_FILE, encoding="utf-8") as f:
        report = json.load(f)
    return format_summary(report), format_rows(report)


# ============ Tab 1：智能问答 ============

def respond(message: str, history: list) -> tuple:
    """多轮对话：history 为之前的对话（list[dict]，纯字符串），返回 (清空输入, 新历史, 引用资料)。"""
    history = list(history or [])
    answer, source_md = kb.chat(message, history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history, source_md


def clear_chat():
    return "", [], "（暂无引用资料。回答知识库问题时会在下方展示参考的原文片段。）"


# ============ Tab 2：知识库管理 ============

def add_docs(filepaths) -> str:
    """把上传的文档加入知识库。filepaths 是绝对路径字符串（列表）。"""
    if not filepaths:
        return "请先选择要上传的文件。"
    if isinstance(filepaths, (str, os.PathLike)):
        filepaths = [filepaths]
    lines = []
    total = 0
    for fp in filepaths:
        fp = str(fp)
        name = os.path.basename(fp)
        n = kb.add_document(fp, name)
        total += n
        lines.append(f"- **{name}**：加入 {n} 块")
    lines.append(f"\n**本次共加入 {total} 块。** 现在可以去「智能问答」里提问新文档的内容。")
    return "\n".join(lines)


def refresh_sources() -> str:
    """返回知识库来源统计的 Markdown 表格。"""
    counts = kb.list_sources()
    if not counts:
        return "知识库目前为空。"
    lines = ["| 来源 | 块数 |", "|---|---|"]
    for src, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {src} | {n} |")
    return "\n".join(lines)


# ============ Tab 3：量化评测 ============

def run_eval(progress=gr.Progress()) -> tuple:
    """重跑评测（走完整 Agent，较慢），返回 (汇总 Markdown, 明细表格)。"""
    def on_progress(i, n, question):
        progress(i / n, desc=f"评测中 {i}/{n}：{question}")

    progress(0, desc="初始化评测…")
    report = evaluate.main(on_progress)
    progress(1.0, desc="评测完成")
    return format_summary(report), format_rows(report)


# ============ 界面搭建 ============

with gr.Blocks(title="企业知识库问答 Agent") as demo:
    gr.Markdown(TOP_INFO)

    with gr.Tabs():
        # ---- Tab 1：智能问答 ----
        with gr.Tab("智能问答"):
            chatbot = gr.Chatbot(height=420, label="对话")
            history_state = gr.State([])
            msg = gr.Textbox(placeholder="输入问题，例如：启明星科技的主打产品是什么？", label="你的问题")
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")
            source_md = gr.Markdown("（暂无引用资料。回答知识库问题时会在下方展示参考的原文片段。）")
            gr.Markdown("> 提示：Agent 会自动调用检索工具从知识库取数，回答会标注来源【资料N】，并支持多轮追问。")

            msg.submit(respond, [msg, history_state], [msg, history_state, chatbot, source_md])
            send_btn.click(respond, [msg, history_state], [msg, history_state, chatbot, source_md])
            clear_btn.click(clear_chat, outputs=[msg, history_state, chatbot, source_md])

        # ---- Tab 2：知识库管理 ----
        with gr.Tab("知识库管理"):
            gr.Markdown("### 上传文档\n选择本地的 **.txt / .md** 文件，将自动切块、向量化并加入知识库（保留来源名）。")
            file_input = gr.File(
                label="上传知识文档（可多选）",
                file_count="multiple",
                file_types=[".txt", ".md"],
            )
            upload_btn = gr.Button("加入知识库", variant="primary")
            upload_status = gr.Markdown("（尚未上传文件）")

            gr.Markdown("### 来源统计")
            refresh_btn = gr.Button("刷新来源统计")
            source_stats = gr.Markdown()

            upload_btn.click(add_docs, inputs=file_input, outputs=upload_status)
            refresh_btn.click(refresh_sources, outputs=source_stats)

        # ---- Tab 3：量化评测 ----
        with gr.Tab("量化评测"):
            gr.Markdown(
                "### 评测指标\n"
                "- **检索命中率**：期望关键词是否全部出现在 top_k 检索结果里（测 RAG 检索质量）\n"
                "- **回答正确率**：Agent 生成的回答是否包含所有期望关键词（测端到端回答质量）\n"
                "- **关键词召回率**：所有关键词的命中比例\n\n"
                "评测集为 `eval/qa_pairs.json`，结果写入 `eval/report.json`。"
            )
            _report = load_report()
            summary_md = gr.Markdown(value=_report[0])
            detail_df = gr.Dataframe(
                value=_report[1],
                headers=["问题", "期望关键词", "检索命中", "检索关键词", "回答正确", "回答关键词"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
            )
            run_btn = gr.Button("运行评测（走完整 Agent，约几分钟）", variant="primary")

            run_btn.click(run_eval, outputs=[summary_md, detail_df])


if __name__ == "__main__":
    kb.ensure_index()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
