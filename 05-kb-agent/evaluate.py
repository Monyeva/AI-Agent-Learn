"""
量化评测：检索命中率 + 回答正确率

指标定义：
- 检索命中率：对每个问题，期望关键词是否全部出现在 top_k 检索结果里（测 RAG 检索质量，快、确定性）
- 回答正确率：Agent 生成的回答是否包含所有期望关键词（测端到端回答质量，慢，走完整 Agent）

用法：python evaluate.py
结果：逐条明细 + 汇总指标，写入 eval/report.json
"""

import json
import os
import time

import kb_agent as kb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QA_FILE = os.path.join(BASE_DIR, "eval", "qa_pairs.json")
REPORT_FILE = os.path.join(BASE_DIR, "eval", "report.json")
# 与 Agent 检索参数保持一致（search_knowledge_base 用的是 top_k=5）
TOP_K = 5


def load_qa_pairs() -> list:
    with open(QA_FILE, encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    """统一字符变体 + 去空白，再做关键词匹配。

    1. 去空白：资料原文和模型回答里会出现「2 亿元」「Agent 平台」这类带空格的写法，
       而期望关键词是「2亿」「Agent平台」，直接子串匹配会因空格误判为不命中。
    2. 字符变体：模型回答常用「7×24」（乘号 ×），而关键词是「7x24」（字母 x），
       × → x 归一化后消除这类误判。
    """
    s = text.replace("×", "x").replace("X", "x")
    return "".join(s.split())


def all_keywords_hit(text: str, keywords: list) -> bool:
    text = normalize(text)
    return all(normalize(kw) in text for kw in keywords)


def metric_retrieval(qa: dict) -> dict:
    """检索命中率：期望关键词是否全部出现在 top_k 检索结果里。"""
    results = kb.retrieve(qa["question"], top_k=TOP_K)
    all_text = " ".join(doc for _, doc, _ in results)
    hit_kws = [kw for kw in qa["expected_keywords"] if normalize(kw) in normalize(all_text)]
    return {
        "hit": len(hit_kws) == len(qa["expected_keywords"]),
        "keyword_hits": len(hit_kws),
        "keyword_total": len(qa["expected_keywords"]),
    }


def metric_answer(qa: dict) -> dict:
    """回答正确率：Agent 生成的回答是否包含所有期望关键词。"""
    answer, _ = kb.chat(qa["question"], [])
    hit_kws = [kw for kw in qa["expected_keywords"] if normalize(kw) in normalize(answer)]
    return {
        "hit": len(hit_kws) == len(qa["expected_keywords"]),
        "keyword_hits": len(hit_kws),
        "keyword_total": len(qa["expected_keywords"]),
    }


def main(progress_cb=None) -> dict:
    """运行评测，写入 eval/report.json，返回报告 dict。

    progress_cb: 可选回调，签名 progress_cb(current_index, total, question_text)，
    用于在 Web 界面里显示逐条进度（命令行运行时传 None 即可）。
    """
    kb.ensure_index()
    qa_list = load_qa_pairs()

    rows = []
    ret_hits = 0
    ans_hits = 0
    kw_hits_total = 0
    kw_total = 0

    print(f"开始评测 {len(qa_list)} 条 QA 对...\n")
    for i, qa in enumerate(qa_list, 1):
        if progress_cb:
            progress_cb(i, len(qa_list), qa["question"])
        # 检索（确定性，快）
        ret = metric_retrieval(qa)
        ret_hits += 1 if ret["hit"] else 0
        kw_hits_total += ret["keyword_hits"]
        kw_total += ret["keyword_total"]

        # 回答（走完整 Agent，慢）
        ans = metric_answer(qa)
        ans_hits += 1 if ans["hit"] else 0

        print(f"[{i}/{len(qa_list)}] {qa['question']}")
        print(f"    检索命中: {'✔' if ret['hit'] else '✘'} ({ret['keyword_hits']}/{ret['keyword_total']}) "
              f"| 回答正确: {'✔' if ans['hit'] else '✘'} ({ans['keyword_hits']}/{ans['keyword_total']})")

        rows.append({
            "question": qa["question"],
            "expected_keywords": qa["expected_keywords"],
            "retrieval_hit": ret["hit"],
            "retrieval_keywords": f"{ret['keyword_hits']}/{ret['keyword_total']}",
            "answer_correct": ans["hit"],
            "answer_keywords": f"{ans['keyword_hits']}/{ans['keyword_total']}",
        })

    n = len(qa_list)
    report = {
        "top_k": TOP_K,
        "total_questions": n,
        "retrieval_hit_rate": round(ret_hits / n, 3),
        "answer_accuracy": round(ans_hits / n, 3),
        "keyword_recall": round(kw_hits_total / kw_total, 3) if kw_total else None,
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"总题数：{n}")
    print(f"检索命中率：{ret_hits}/{n} = {report['retrieval_hit_rate']:.1%}")
    print(f"回答正确率：{ans_hits}/{n} = {report['answer_accuracy']:.1%}")
    print(f"关键词召回率：{report['keyword_recall']:.1%}")
    print(f"报告已写入：{REPORT_FILE}")
    return report


if __name__ == "__main__":
    main()
