"""RAG evaluation metrics, computed without any extra model or API key.

Retrieval quality (did we fetch the right text?)
  hit_rate@k        expected source file is among the retrieved sources
  MRR               1 / rank of the first retrieved source from the expected file
  precision@k       share of retrieved sources that come from the expected file
  context_recall    the answer keyword is present in the retrieved excerpts

Generation quality (is the answer right and safe?)
  answer_accuracy   answer contains an expected keyword (or correctly refuses)
  faithfulness      mean grounding score: share of answer words/numbers supported by the excerpts
  grounded_rate     share of answers that pass the hallucination check
  refusal_accuracy  out-of-scope questions are refused
  false_refusal     answerable questions wrongly refused

Efficiency: latency p50/p95, tokens per question, cache hit rate.

These are keyword-and-overlap metrics: fast, free and reproducible, but not as nuanced as an
LLM judge such as RAGAS or DeepEval (both open source, they need a judge model).
"""
import numpy as np

from src import config


def _keywords(row: dict) -> list[str]:
    return [k.strip().lower() for k in row.get("keywords", "").split("|") if k.strip()]


def is_refusal(row: dict) -> bool:
    return str(row.get("expect_refusal", "0")).strip() == "1"


def retrieval_metrics(row: dict, sources: list[dict], contexts: list[str]) -> dict:
    """Metrics for one question. Not applicable (None) for out-of-scope questions."""
    if is_refusal(row):
        return {"hit": None, "reciprocal_rank": None, "precision": None, "context_recall": None}
    expected = row.get("source_file", "").strip()
    files = [s["source"] for s in sources]
    hit = expected in files
    rr = 1 / (files.index(expected) + 1) if hit else 0.0
    precision = files.count(expected) / len(files) if files else 0.0
    kws = _keywords(row)
    joined = " ".join(" ".join(contexts).lower().split())
    recall = (any(k in joined for k in kws) if kws else True)
    return {"hit": bool(hit), "reciprocal_rank": rr, "precision": precision, "context_recall": bool(recall)}


def answer_correct(row: dict, answer: str) -> bool:
    text = answer.lower()
    if is_refusal(row):
        return "could not find" in text
    kws = _keywords(row)
    return any(k in text for k in kws) if kws else True


def summarize(records: list[dict]) -> dict:
    """records: one dict per question with keys from evaluate.py."""
    answerable = [r for r in records if not r["expect_refusal"]]
    refusals = [r for r in records if r["expect_refusal"]]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    lat = [r["latency_ms"] for r in records]
    tok = [r["total_tokens"] for r in records]
    return {
        "questions": len(records),
        "retrieval": {
            "hit_rate_at_k": mean([r["hit"] for r in answerable]),
            "mrr": mean([r["reciprocal_rank"] for r in answerable]),
            "precision_at_k": mean([r["precision"] for r in answerable]),
            "context_recall": mean([r["context_recall"] for r in answerable]),
            "k": config.TOP_K,
        },
        "generation": {
            "answer_accuracy": mean([r["correct"] for r in records]),
            "faithfulness": mean([r["grounding_score"] for r in records]),
            "grounded_rate": mean([r["grounded"] for r in records]),
            "refusal_accuracy": mean([r["correct"] for r in refusals]),
            "false_refusal_rate": mean(
                [config.NOT_FOUND_ANSWER in r["answer"] for r in answerable]
            ),
        },
        "efficiency": {
            "latency_p50_ms": int(np.percentile(lat, 50)),
            "latency_p95_ms": int(np.percentile(lat, 95)),
            "avg_tokens_per_question": int(np.mean(tok)),
            "total_tokens": int(np.sum(tok)),
        },
    }