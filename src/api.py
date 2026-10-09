"""FastAPI service.

Run:  python -m uvicorn src.api:app --port 8000
Docs: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.ingest import build_index
from src.pii import engine_name
from src.rag_chain import build_chatbot
from src.usage import prometheus_text

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads the embedding model, the LLM and the index. The first start downloads the
    # models, so this can take a few minutes; the port opens only after it finishes.
    _state["bot"] = build_chatbot()
    yield
    _state["bot"].tracer.shutdown()
    _state.clear()


app = FastAPI(
    title="Loan Policy RAG API",
    description="Answers loan policy questions from the policy documents using open-source models.",
    version="2.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=3, max_length=500, examples=["What is the maximum tenure of a home loan?"])


class Source(BaseModel):
    source: str
    page: Optional[int] = None
    score: float
    snippet: str


class Tokens(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    latency_ms: int
    request_id: str = ""
    grounded: bool = True                 # hallucination check passed
    grounding_score: float = 1.0
    warnings: list[str] = []
    blocked: str = ""                     # guardrail category if the request was blocked
    pii_masked: dict[str, int] = {}       # what was masked, e.g. {"PAN": 1}
    model: str = ""
    tier: str = ""                        # "simple" (small model) or "complex"
    cache_hit: str = ""                   # "", "exact" or "semantic"
    escalated: bool = False               # small model failed the check, main model re-answered
    tokens: Tokens = Tokens()
    retrieval_ms: int = 0
    generation_ms: int = 0


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "fake_models": config.USE_FAKE_MODELS,
        "routing": config.ROUTING_ENABLED,
        "small_llm": config.LLM_MODEL_SMALL if config.ROUTING_ENABLED else None,
        "cache": config.CACHE_ENABLED,
        "guardrails": config.GUARDRAILS_ENABLED,
        "pii_masking": config.PII_MASKING,
        "pii_engine": engine_name() if config.PII_MASKING else "off",
        "langfuse": getattr(_state.get("bot"), "tracer", None) is not None and _state["bot"].tracer.enabled,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    bot = _state.get("bot")
    if bot is None:
        raise HTTPException(status_code=503, detail="The model is still loading. Try again shortly.")
    return bot.ask(req.question)  # the bot serialises LLM calls itself, cache hits skip the queue


@app.post("/reindex")
def reindex() -> dict:
    """Rebuild the vector index from the files in data/policies and reload it."""
    bot = _state.get("bot")
    if bot is None:
        raise HTTPException(status_code=503, detail="The model is still loading. Try again shortly.")
    with bot.lock:
        pages, chunks = build_index()
        bot.reload()
    return {"status": "reindexed", "pages": pages, "chunks": chunks}


def _tracker():
    bot = _state.get("bot")
    if bot is None:
        raise HTTPException(status_code=503, detail="The model is still loading. Try again shortly.")
    if bot.tracker is None:
        raise HTTPException(status_code=404, detail="Usage tracking is not enabled.")
    return bot.tracker


@app.get("/usage/summary")
def usage_summary(window_hours: Optional[float] = None) -> dict:
    """Tokens, latency, cache hit rate, blocks, PII masked, hallucination flags, estimated cost."""
    return _tracker().summary(since_seconds=window_hours * 3600 if window_hours else None)


@app.get("/usage/recent")
def usage_recent(limit: int = 50) -> list[dict]:
    return _tracker().recent(limit=min(max(limit, 1), 500))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text format, so Prometheus/Grafana can scrape the service."""
    return prometheus_text(_tracker().summary())