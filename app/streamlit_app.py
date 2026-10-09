"""Streamlit chat UI. Talks to the FastAPI service over HTTP.

Run:  python -m streamlit run app/streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

EXAMPLES = [
    "What is the minimum monthly income for a salaried personal loan applicant?",
    "What is the maximum loan to value ratio for a Rs. 50 lakh home loan?",
    "What is the foreclosure charge if I close a personal loan after 8 months?",
    "After how many days of overdue is an account classified as NPA?",
    "Are there prepayment charges on floating rate home loans?",
]

st.set_page_config(page_title="Loan Policy Assistant", page_icon="🏦", layout="wide")


def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def ask_api(question: str) -> dict:
    r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=300)
    r.raise_for_status()
    return r.json()


def get_json(path: str, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def badges(result: dict) -> str:
    """One-line summary of what the pipeline did for this answer."""
    parts = []
    if result.get("blocked"):
        parts.append(f"blocked: {result['blocked']}")
    if result.get("cache_hit"):
        parts.append(f"cache hit ({result['cache_hit']})")
    elif result.get("model") and not result.get("blocked"):
        parts.append(f"model: {result['model']}" + (" (escalated)" if result.get("escalated") else ""))
        t = result.get("tokens", {})
        parts.append(f"{t.get('total_tokens', 0)} tokens")
    if result.get("pii_masked"):
        parts.append("PII masked: " + ", ".join(f"{k}x{v}" for k, v in result["pii_masked"].items()))
    if not result.get("blocked"):
        parts.append("grounded" if result.get("grounded", True) else "NOT verified")
    parts.append(f"{result['latency_ms'] / 1000:.1f} s")
    return " | ".join(parts)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = f", page {s['page']}" if s.get("page") else ""
            st.markdown(f"**{s['source']}{page}** (similarity {s['score']})")
            st.caption(s["snippet"] + " ...")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Loan Policy Assistant")
    st.caption("Answers come only from the loan policy documents, using open-source models running locally.")
    health = api_health()
    if health:
        st.success("API connected")
        st.caption(f"LLM: {health['llm']}\n\nEmbeddings: {health['embedding_model']}")
        on = lambda b: "on" if b else "off"
        st.caption(
            f"Routing: {on(health.get('routing'))} | Cache: {on(health.get('cache'))} | "
            f"Guardrails: {on(health.get('guardrails'))} | PII masking: {on(health.get('pii_masking'))} | "
            f"Langfuse: {on(health.get('langfuse'))}"
        )
    else:
        st.error(f"API not reachable at {API_URL}")
    st.divider()
    page = st.radio("Page", ["Chat", "Monitoring"], horizontal=True, label_visibility="collapsed")
    st.divider()
    st.subheader("Try an example")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"ex{i}", width="stretch"):
            st.session_state["pending"] = example
    st.divider()
    if st.button("Clear chat", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()

# ---------- monitoring ----------
if page == "Monitoring":
    st.title("Monitoring: tokens, cost, latency, safety")
    window = st.selectbox("Time window", ["All time", "Last 24 hours", "Last hour"])
    hours = {"All time": None, "Last 24 hours": 24, "Last hour": 1}[window]
    summary = get_json("/usage/summary", **({"window_hours": hours} if hours else {}))
    if not summary:
        st.warning("No usage data yet, or the API is not reachable.")
        st.stop()

    c = st.columns(4)
    c[0].metric("Questions", summary["requests"])
    c[1].metric("LLM calls", summary["llm_calls"])
    c[2].metric("Cache hit rate", f"{summary['cache_hit_rate']:.0%}")
    c[3].metric("Avg latency", f"{summary['avg_latency_ms'] / 1000:.1f} s")
    c = st.columns(4)
    c[0].metric("Total tokens", f"{summary['total_tokens']:,}")
    c[1].metric("Tokens saved by cache", f"{summary['tokens_saved_by_cache']:,}")
    c[2].metric("p95 latency", f"{summary['p95_latency_ms'] / 1000:.1f} s")
    c[3].metric("Real API cost", f"${summary['actual_api_cost_usd']:.2f}")
    st.caption(
        f"Equivalent cost if the same traffic ran on a paid API (assumed prices, editable in config): "
        f"${summary['est_equivalent_api_cost_usd']:.4f}. Saved by the cache: ${summary['est_saved_by_cache_usd']:.4f}."
    )

    st.subheader("Safety")
    c = st.columns(4)
    c[0].metric("Blocked by guardrails", summary["blocked"])
    c[1].metric("Requests with PII masked", summary["pii_masked_requests"])
    c[2].metric("Ungrounded answers", summary["ungrounded_answers"])
    c[3].metric("Escalated to main model", summary["escalations"])

    left, right = st.columns(2)
    with left:
        st.subheader("Tokens by model")
        if summary["by_model"]:
            st.bar_chart({m: v["tokens"] for m, v in summary["by_model"].items()})
        else:
            st.caption("No LLM calls yet.")
    with right:
        st.subheader("Requests by routing tier")
        if summary["requests_by_tier"]:
            st.bar_chart(summary["requests_by_tier"])
        else:
            st.caption("No data yet.")

    st.subheader("Recent requests (questions are stored with PII masked)")
    recent = get_json("/usage/recent", limit=50) or []
    if recent:
        cols = ["ts", "question", "model", "tier", "total_tokens", "latency_ms", "cache_hit", "blocked", "pii_types", "grounded"]
        st.dataframe([{k: r.get(k) for k in cols} for r in recent], width="stretch")
    st.stop()

# ---------- chat ----------
st.title("Loan Policy Q&A")
st.session_state.setdefault("messages", [])

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            for w in message.get("warnings", []):
                st.warning(w)
            show_sources(message.get("sources", []))
            if message.get("badges"):
                st.caption(message["badges"])

typed = st.chat_input("Ask about eligibility, fees, prepayment, documents, overdue rules...")
question = st.session_state.pop("pending", None) or typed

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the policy documents..."):
                result = ask_api(question)
            st.markdown(result["answer"])
            for w in result.get("warnings", []):
                st.warning(w)
            show_sources(result["sources"])
            summary_line = badges(result)
            st.caption(summary_line)
            st.session_state["messages"].append(
                {
                    "role": "assistant", "content": result["answer"], "sources": result["sources"],
                    "warnings": result.get("warnings", []), "badges": summary_line,
                }
            )
        except requests.RequestException as exc:
            st.error(f"Could not get an answer from the API: {exc}")