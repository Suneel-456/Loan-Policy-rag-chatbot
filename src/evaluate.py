"""Evaluate the chatbot on data/eval/test_questions.csv and report RAG metrics.

Run:   python -m src.evaluate            (use --min-pass 0.7 to fail CI below 70% answer accuracy)
Writes reports/eval_results.csv (per question) and reports/eval_summary.json (metrics).
The cache is bypassed so latencies are real. Read the answers too, not just the scores.
"""
import argparse
import csv
import json
import sys

from src import config
from src.rag_chain import build_chatbot
from src.rag_metrics import answer_correct, is_refusal, retrieval_metrics, summarize


def run_evaluation(bot, rows: list[dict]) -> tuple[list[dict], dict]:
    records = []
    for row in rows:
        out = bot.ask(row["question"], use_cache=False, track=False)
        ret = retrieval_metrics(row, out["sources"], out["contexts"])
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected": row["expected_answer"],
                "answer": out["answer"],
                "expect_refusal": is_refusal(row),
                "correct": answer_correct(row, out["answer"]),
                "grounded": out["grounded"],
                "grounding_score": out["grounding_score"],
                "top_source": out["sources"][0]["source"] if out["sources"] else "",
                "model": out["model"],
                "total_tokens": out["tokens"]["total_tokens"],
                "latency_ms": out["latency_ms"],
                **ret,
            }
        )
    return records, summarize(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the loan policy chatbot")
    parser.add_argument("--min-pass", type=float, default=0.0, help="exit 1 if answer accuracy is below this (0-1)")
    args = parser.parse_args()

    with open(config.EVAL_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records, summary = run_evaluation(build_chatbot(), rows)
    for r in records:
        print(f"[{'PASS' if r['correct'] else 'FAIL'}] Q{r['id']}: {r['question']}")
        print(f"       expected: {r['expected']}")
        print(f"       answer  : {r['answer']}")
        print(f"       top source: {r['top_source'] or '-'} | grounded: {r['grounded']} | {r['latency_ms']} ms | {r['total_tokens']} tokens")

    print("\n=== RAG metrics ===")
    for group in ("retrieval", "generation", "efficiency"):
        print(f"{group}:")
        for k, v in summary[group].items():
            print(f"  {k:26s} {v}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (config.REPORTS_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {config.REPORTS_DIR / 'eval_results.csv'} and eval_summary.json")

    accuracy = summary["generation"]["answer_accuracy"] or 0.0
    return 0 if accuracy >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())