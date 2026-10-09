"""The RAG pipeline.

question -> PII mask -> input guardrails -> cache -> retrieve -> route to a model -> generate
         -> grounding (hallucination) check [-> escalate to the bigger model] -> output guardrails
         -> cache + usage log + optional Langfuse trace -> answer with sources
"""
import threading
import time
import uuid

from langchain_core.prompts import PromptTemplate

from src import config
from src.cache import AnswerCache
from src.embeddings import get_embeddings
from src.grounding import check_grounding
from src.guardrails import check_input, check_output
from src.ingest import build_index
from src.llm import get_llm, get_small_llm
from src.observability import NullTracer, get_tracer
from src.pii import mask_pii, warm_up as warm_up_pii
from src.retriever import load_store, retrieve
from src.routing import ModelRouter
from src.usage import UsageTracker, estimate_cost

TEMPLATE = """You are a loan policy assistant for Sample Bank.
Answer the question using ONLY the policy excerpts below.
If the answer is not in the excerpts, reply exactly: "__NOT_FOUND__"
Be concise (one to three sentences). Quote numbers, percentages and durations exactly as written.
Mention the section number when the excerpt shows one.

Policy excerpts:
{context}

Question: {question}
Answer:""".replace("__NOT_FOUND__", config.NOT_FOUND_ANSWER)

PROMPT = PromptTemplate.from_template(TEMPLATE)


def format_context(results, max_chars: int = None) -> str:
    """Numbered excerpts. Stops adding excerpts once max_chars is reached (shorter prompt =
    faster generation), but always keeps at least the best one."""
    max_chars = config.MAX_CONTEXT_CHARS if max_chars is None else max_chars
    parts, used = [], 0
    for i, (doc, _score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source}, page {page}" if page else source
        part = f"[{i}] ({label})\n{doc.page_content.strip()}"
        if parts and used + len(part) > max_chars:
            break
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)


def format_sources(results) -> list[dict]:
    """One entry per (file, page), keeping the best score."""
    best: dict[tuple, dict] = {}
    for doc, score in results:
        key = (doc.metadata.get("source", "unknown"), doc.metadata.get("page"))
        if key not in best or score > best[key]["score"]:
            best[key] = {
                "source": key[0],
                "page": key[1],
                "score": round(score, 3),
                "snippet": " ".join(doc.page_content.split())[:220],
            }
    return sorted(best.values(), key=lambda s: s["score"], reverse=True)


def _empty_tokens() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class LoanPolicyBot:
    def __init__(
        self,
        store,
        embeddings,
        llm,
        k: int = None,
        min_score: float = None,
        router: ModelRouter = None,
        cache: AnswerCache = None,
        tracker: UsageTracker = None,
        tracer=None,
    ):
        self.store = store
        self.embeddings = embeddings
        self.k = config.TOP_K if k is None else k
        self.min_score = config.MIN_RELEVANCE if min_score is None else min_score
        self.router = router or ModelRouter(llm, None, enabled=False)
        self.cache = cache
        self.tracker = tracker
        self.tracer = tracer or NullTracer()
        self.lock = threading.Lock()  # one generation at a time (the local LLM is CPU-bound)

    def reload(self) -> None:
        """Re-read the index from disk (used after re-ingesting documents)."""
        self.store = load_store(self.embeddings)
        if self.cache is not None:
            self.cache.clear()  # cached answers may be out of date

    # ------------------------------------------------------------------ public
    def ask(self, question: str, use_cache: bool = True, track: bool = True) -> dict:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        safe_q, pii = mask_pii(question) if config.PII_MASKING else (question, {})

        with self.tracer.observe("loan-rag-ask", as_type="span", input=safe_q) as root:
            result = self._answer(safe_q, pii, use_cache, request_id)
            result["question"] = question
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            root.update(
                output=result["answer"],
                metadata={
                    "request_id": request_id, "model": result["model"], "tier": result["tier"],
                    "cache_hit": result["cache_hit"], "blocked": result["blocked"],
                    "grounded": result["grounded"], "grounding_score": result["grounding_score"],
                    "tokens": result["tokens"], "pii_masked": result["pii_masked"],
                },
            )

        if track and self.tracker is not None:
            t = result["tokens"]
            self.tracker.record(
                request_id=request_id, question=safe_q, model=result["model"], tier=result["tier"],
                prompt_tokens=t["prompt_tokens"], completion_tokens=t["completion_tokens"],
                total_tokens=t["total_tokens"], latency_ms=result["latency_ms"],
                retrieval_ms=result["retrieval_ms"], generation_ms=result["generation_ms"],
                cache_hit=result["cache_hit"], saved_tokens=result.get("saved_tokens", 0),
                blocked=result["blocked"], pii_types=",".join(sorted(result["pii_masked"])),
                grounded=None if result["blocked"] else int(result["grounded"]),
                grounding_score=result["grounding_score"], escalated=int(result["escalated"]),
                est_cost_usd=estimate_cost(t["prompt_tokens"], t["completion_tokens"]),
            )
        return result

    # ----------------------------------------------------------------- internal
    def _base(self, request_id: str, pii: dict) -> dict:
        return {
            "request_id": request_id, "answer": "", "sources": [], "contexts": [],
            "grounded": True, "grounding_score": 1.0, "warnings": [], "blocked": "",
            "pii_masked": dict(pii), "model": "", "tier": "", "cache_hit": "",
            "escalated": False, "tokens": _empty_tokens(), "saved_tokens": 0,
            "retrieval_ms": 0, "generation_ms": 0,
        }

    def _generate(self, route, prompt: str) -> tuple[str, dict, int]:
        """One LLM call under the lock. Returns (answer, usage, milliseconds)."""
        with self.tracer.observe("generation", as_type="generation", model=route.model, input=prompt) as gen:
            with self.lock:
                t0 = time.perf_counter()
                answer = route.llm.invoke(prompt).strip()
                ms = int((time.perf_counter() - t0) * 1000)
                usage = getattr(route.llm, "last_usage", {"prompt_tokens": 0, "completion_tokens": 0})
            gen.update(
                output=answer,
                usage_details={"input": usage["prompt_tokens"], "output": usage["completion_tokens"]},
            )
        return answer, usage, ms

    def _answer(self, safe_q: str, pii: dict, use_cache: bool, request_id: str) -> dict:
        out = self._base(request_id, pii)

        # 1) input guardrails
        if config.GUARDRAILS_ENABLED:
            with self.tracer.observe("input-guardrail", as_type="guardrail", input=safe_q) as g:
                verdict = check_input(safe_q)
                g.update(output={"allowed": verdict.allowed, "category": verdict.category})
            if not verdict.allowed:
                out.update(answer=verdict.message, blocked=verdict.category, model="guardrail")
                return out

        # 2) embed once; reuse for cache lookup and retrieval
        vector = self.embeddings.embed_query(safe_q)

        # 3) cache
        if self.cache is not None and use_cache and config.CACHE_ENABLED:
            hit = self.cache.get(safe_q, vector)
            if hit:
                cached, kind = hit
                out = {**cached, "request_id": request_id, "cache_hit": kind,
                       "pii_masked": dict(pii), "tokens": _empty_tokens(),
                       "saved_tokens": cached["tokens"]["total_tokens"] or cached.get("saved_tokens", 0),
                       "retrieval_ms": 0, "generation_ms": 0, "escalated": False}
                return out

        # 4) retrieval
        with self.tracer.observe("retrieval", as_type="retriever", input=safe_q) as r:
            t0 = time.perf_counter()
            results = retrieve(self.store, safe_q, self.k, self.min_score, query_vector=vector)
            out["retrieval_ms"] = int((time.perf_counter() - t0) * 1000)
            r.update(output=[d.metadata.get("source") for d, _ in results])
        out["sources"] = format_sources(results)
        out["contexts"] = [d.page_content for d, _ in results]
        if not results:
            out["answer"] = config.NOT_FOUND_ANSWER
            out["model"] = "none (no relevant excerpts)"
            return out

        # 5) route + generate
        context = format_context(results)
        prompt = PROMPT.format(context=context, question=safe_q)
        route = self.router.route(safe_q)
        answer, usage, ms = self._generate(route, prompt)
        used = dict(usage)
        out["generation_ms"] = ms
        out["tier"], out["model"] = route.tier, route.model

        # 6) hallucination (grounding) check, with escalation from the small to the main model
        # the numbered/labelled prompt context is included so citations like "[1]" or "page 2" count as supported
        full_context = context + "\n\n" + "\n\n".join(out["contexts"])
        grounding = check_grounding(answer, full_context)
        if not grounding.grounded and self.router.enabled and route.tier == "simple":
            big = self.router.main_route()
            answer, usage2, ms2 = self._generate(big, prompt)
            used = {k: used[k] + usage2[k] for k in used}
            out.update(generation_ms=ms + ms2, escalated=True, tier="complex", model=big.model)
            grounding = check_grounding(answer, full_context)

        # 7) output guardrails (prompt leak, PII in the answer)
        if config.GUARDRAILS_ENABLED:
            answer, verdict, pii_out = check_output(answer)
            for k, v in pii_out.items():
                out["pii_masked"][k] = out["pii_masked"].get(k, 0) + v
            if not verdict.allowed:
                out.update(answer=answer, blocked=verdict.category)
                out["tokens"] = {**used, "total_tokens": used["prompt_tokens"] + used["completion_tokens"]}
                return out

        out["grounded"], out["grounding_score"] = grounding.grounded, grounding.score
        if not grounding.grounded:
            detail = f" (numbers not found in the policy text: {', '.join(grounding.ungrounded_numbers)})" \
                if grounding.ungrounded_numbers else ""
            if config.HALLUCINATION_ACTION == "block":
                answer = config.NOT_FOUND_ANSWER
                out["warnings"].append("The generated answer could not be verified against the policy text and was withheld" + detail)
            else:
                out["warnings"].append("This answer could not be fully verified against the policy text" + detail + ". Please check the sources.")
        out["answer"] = answer
        out["tokens"] = {**used, "total_tokens": used["prompt_tokens"] + used["completion_tokens"]}

        # 8) cache only verified answers
        if self.cache is not None and use_cache and config.CACHE_ENABLED and grounding.grounded:
            self.cache.put(safe_q, {**out, "pii_masked": {}}, vector)
        return out


def build_chatbot() -> LoanPolicyBot:
    """Build the index on first run if needed, then load models and the index."""
    if not config.VECTORSTORE_PATH.exists():
        build_index()
    embeddings = get_embeddings()
    llm = get_llm()
    if config.PII_MASKING:
        warm_up_pii()  # loads Presidio now so the first question is not slow
    return LoanPolicyBot(
        load_store(embeddings),
        embeddings,
        llm,
        router=ModelRouter(llm, get_small_llm),
        cache=AnswerCache() if config.CACHE_ENABLED else None,
        tracker=UsageTracker(),
        tracer=get_tracer(),
    )