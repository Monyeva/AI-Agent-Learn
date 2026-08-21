"""
办公自动化 Agent —— 命令行多轮对话入口

用法：python cli.py
支持多轮对话（携带上文记忆），输入 exit 或 quit 退出。
"""

import office_agent as oa

WELCOME = """办公自动化 Agent（命令行版）
能：读/写 Excel、查数据库、发邮件、生成报表。
输入你的任务，或输入 exit 退出。
示例：
  - 华东地区的总销售额是多少？
  - 技术部的平均薪资是多少？
  - 把每个地区的销售额汇总成 Excel 报表
  - 给张伟发一封邮件，提醒他提交市场周报
"""


def main():
    print(WELCOME)
    history = []
    while True:
        try:
            question = input("\n你：\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break

        print("\nAgent：")
        answer = oa.chat(question, history)
        print(answer)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
