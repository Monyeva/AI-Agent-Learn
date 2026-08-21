"""
多 Agent 调研报告生成器 —— 命令行多轮对话入口

用法：
    python cli.py

示例指令：
- 写一份启明星科技的市场调研报告
- 分析一下华东地区的销售情况，出个报告
- 调研公司融资和团队，写一份背景报告
输入 exit 退出。支持多轮对话（携带上文记忆）。
"""

import os
import sys

# 允许运行时通过命令行临时设置，避免硬编码 key：
#   python cli.py --key sk-xxx --base-url http://... --model deepseek-v4-pro
# 或通过环境变量 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL
def _parse_args():
    args = sys.argv[1:]
    out = {}
    for i, a in enumerate(args):
        if a in ("--key", "--base-url", "--model") and i + 1 < len(args):
            out[a.lstrip("-").replace("-", "_")] = args[i + 1]
    return out

_cli = _parse_args()
if _cli.get("key"):
    os.environ["OPENAI_API_KEY"] = _cli["key"]
if _cli.get("base_url"):
    os.environ["OPENAI_BASE_URL"] = _cli["base_url"]
if _cli.get("model"):
    os.environ["MODEL"] = _cli["model"]

import report_agent as ra


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "⚠️  未检测到 OPENAI_API_KEY 环境变量。\n"
            "    运行方式：\n"
            "    1) python cli.py --key sk-xxx [--base-url http://...] [--model ...]\n"
            "    2) 或先设置环境变量：set OPENAI_API_KEY=sk-xxx"
        )
        print()

    history = []
    print("多 Agent 调研报告生成器（Supervisor 模式）")
    print("输入调研报告主题，或直接回车看示例。输入 exit 退出。\n")

    while True:
        try:
            question = input("你：\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if question.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break
        if not question:
            question = "调研启明星科技的市场表现，写一份 2024 年市场调研报告"
            print(f"（使用示例任务：{question}）")

        print("\n🔄 多 Agent 协作中（Supervisor 派发研究员 + 报告撰写员）...\n")
        answer = ra.chat(question, history)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        print(f"结果：\n{answer}\n" + "-" * 60)


if __name__ == "__main__":
    main()
