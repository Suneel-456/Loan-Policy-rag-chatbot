"""Usage tracking in SQLite (standard library only, no server): tokens, latency, cache hits,
guardrail blocks, PII masked, hallucination flags, routing, and estimated equivalent cost.
Only MASKED questions are stored. Read it through GET /usage/summary, GET /usage/recent,
GET /metrics (Prometheus format) or the Streamlit "Monitoring" tab.
"""
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src import config

_COLUMNS = [
    ("request_id", "TEXT"), ("ts", "REAL"), ("question", "TEXT"), ("model", "TEXT"), ("tier", "TEXT"),
    ("prompt_tokens", "INTEGER"), ("completion_tokens", "INTEGER"), ("total_tokens", "INTEGER"),
    ("latency_ms", "INTEGER"), ("retrieval_ms", "INTEGER"), ("generation_ms", "INTEGER"),
    ("cache_hit", "TEXT"), ("saved_tokens", "INTEGER"), ("blocked", "TEXT"), ("pii_types", "TEXT"),
    ("grounded", "INTEGER"), ("grounding_score", "REAL"), ("escalated", "INTEGER"),
    ("est_cost_usd", "REAL"),
]
_lock = threading.Lock()


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * config.PRICE_PER_1M_INPUT_USD + completion_tokens * config.PRICE_PER_1M_OUTPUT_USD
    ) / 1_000_000


class UsageTracker:
    def __init__(self, path: Path = None):
        self.path = Path(config.USAGE_DB_PATH if path is None else path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cols = ", ".join(f"{n} {t}" for n, t in _COLUMNS)
        with self._conn() as c:
            c.execute(f"CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def record(self, **row) -> None:
        row.setdefault("ts", time.time())
        names = [n for n, _ in _COLUMNS]
        values = [row.get(n) for n in names]
        with _lock, self._conn() as c:
            c.execute(
                f"INSERT INTO calls ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})", values
            )

    def recent(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def summary(self, since_seconds: Optional[float] = None) -> dict:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            q = "SELECT * FROM calls"
            args: tuple = ()
            if since_seconds:
                q += " WHERE ts >= ?"
                args = (time.time() - since_seconds,)
            rows = [dict(r) for r in c.execute(q, args).fetchall()]

        n = len(rows)
        llm_rows = [r for r in rows if not r["cache_hit"] and not r["blocked"] and r["total_tokens"]]
        lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        by_model: dict[str, dict] = {}
        for r in llm_rows:
            m = by_model.setdefault(r["model"] or "unknown", {"requests": 0, "tokens": 0})
            m["requests"] += 1
            m["tokens"] += r["total_tokens"]
        by_tier: dict[str, int] = {}
        for r in rows:
            if r["tier"]:
                by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        blocked: dict[str, int] = {}
        for r in rows:
            if r["blocked"]:
                blocked[r["blocked"]] = blocked.get(r["blocked"], 0) + 1

        prompt = sum(r["prompt_tokens"] or 0 for r in rows)
        completion = sum(r["completion_tokens"] or 0 for r in rows)
        saved = sum(r["saved_tokens"] or 0 for r in rows)
        hits = sum(1 for r in rows if r["cache_hit"])
        return {
            "requests": n,
            "llm_calls": len(llm_rows),
            "cache_hits": hits,
            "cache_hit_rate": round(hits / n, 3) if n else 0.0,
            "blocked": sum(blocked.values()),
            "blocked_by_category": blocked,
            "pii_masked_requests": sum(1 for r in rows if r["pii_types"]),
            "ungrounded_answers": sum(1 for r in rows if r["grounded"] == 0),
            "escalations": sum(1 for r in rows if r["escalated"]),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "tokens_saved_by_cache": saved,
            "avg_latency_ms": int(np.mean(lat)) if lat else 0,
            "p50_latency_ms": int(np.percentile(lat, 50)) if lat else 0,
            "p95_latency_ms": int(np.percentile(lat, 95)) if lat else 0,
            "requests_by_tier": by_tier,
            "by_model": by_model,
            "est_equivalent_api_cost_usd": round(sum(r["est_cost_usd"] or 0 for r in rows), 6),
            "est_saved_by_cache_usd": round(saved * config.PRICE_PER_1M_OUTPUT_USD / 1_000_000, 6),
            "actual_api_cost_usd": 0.0,  # models run locally
        }


def prometheus_text(s: dict) -> str:
    """Render a summary as Prometheus exposition text (scrape with Prometheus / Grafana)."""
    lines = []

    def metric(name: str, value, help_: str, kind: str = "counter"):
        lines.extend([f"# HELP {name} {help_}", f"# TYPE {name} {kind}", f"{name} {value}"])

    metric("loan_rag_requests_total", s["requests"], "Questions received")
    metric("loan_rag_llm_calls_total", s["llm_calls"], "Questions that needed an LLM call")
    metric("loan_rag_cache_hits_total", s["cache_hits"], "Answers served from cache")
    metric("loan_rag_blocked_total", s["blocked"], "Questions blocked by guardrails")
    metric("loan_rag_ungrounded_answers_total", s["ungrounded_answers"], "Answers that failed the grounding check")
    metric("loan_rag_prompt_tokens_total", s["prompt_tokens"], "Prompt tokens processed")
    metric("loan_rag_completion_tokens_total", s["completion_tokens"], "Completion tokens generated")
    metric("loan_rag_latency_avg_ms", s["avg_latency_ms"], "Average end-to-end latency (ms)", "gauge")
    metric("loan_rag_latency_p95_ms", s["p95_latency_ms"], "p95 end-to-end latency (ms)", "gauge")
    return "\n".join(lines) + "\n"