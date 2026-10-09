"""Central configuration. Every value can be overridden with an environment variable,
which is how Docker, docker-compose and Kubernetes configure the app."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
POLICY_DIR = DATA_DIR / "policies"
EVAL_FILE = DATA_DIR / "eval" / "test_questions.csv"
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", BASE_DIR / "vectorstore" / "index.json"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", BASE_DIR / "reports"))

# Open-source models, downloaded from Hugging Face on first run. No API keys needed.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# USE_FAKE_MODELS=1 swaps in tiny stand-in models. Used only by tests and CI,
# so the pipeline can be tested without downloading any model.
USE_FAKE_MODELS = os.getenv("USE_FAKE_MODELS", "0") == "1"

# Chunking and retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))
# Minimum cosine similarity for a chunk to be used. -1 disables the filter.
# Raise it (for example to 0.35) to refuse off-topic questions before the LLM is called.
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "-1.0"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "80"))

NOT_FOUND_ANSWER = "I could not find this in the loan policy documents."

# Lines matching these patterns (page footers and similar) are removed before chunking.
BOILERPLATE_PATTERNS = [
    r"^Sample Bank \(fictional\).*$",
    r"^Page \d+$",
]


# ---------------------------------------------------------------------------
# Safety: PII masking, guardrails, hallucination (grounding) check
# ---------------------------------------------------------------------------
PII_MASKING = os.getenv("PII_MASKING", "1") == "1"
# "regex"    = built-in Indian-ID rules only (PAN, Aadhaar, phone, IFSC, card, account, email)
# "presidio" = regex + Microsoft Presidio (also finds names and places); fails if Presidio is missing
# "auto"     = use Presidio when installed and its spaCy model is available, otherwise regex only
PII_ENGINE = os.getenv("PII_ENGINE", "regex")
PRESIDIO_SPACY_MODEL = os.getenv("PRESIDIO_SPACY_MODEL", "en_core_web_sm")
# Domain terms that Presidio's name detector sometimes mistakes for people. Comma separated.
PII_ALLOW_LIST = [t.strip() for t in os.getenv(
    "PII_ALLOW_LIST", "Loan Amount,Home Loan,Personal Loan,Gold Loan,Sample Bank").split(",") if t.strip()]
GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "1") == "1"
# Grounding check: share of the answer's content words that must appear in the retrieved
# excerpts. Any number in the answer that is NOT in the excerpts always counts as ungrounded.
GROUNDING_MIN_SCORE = float(os.getenv("GROUNDING_MIN_SCORE", "0.6"))
# "flag"  = return the answer with grounded=false and a warning
# "block" = replace an ungrounded answer with the not-found message
HALLUCINATION_ACTION = os.getenv("HALLUCINATION_ACTION", "flag")

# ---------------------------------------------------------------------------
# Cost and latency: model routing, caching, context budget
# ---------------------------------------------------------------------------
# Routing sends simple lookup questions to a small model and complex ones to LLM_MODEL.
# If the small model's answer fails the grounding check, the question is escalated to LLM_MODEL.
ROUTING_ENABLED = os.getenv("ROUTING_ENABLED", "0") == "1"
LLM_MODEL_SMALL = os.getenv("LLM_MODEL_SMALL", "Qwen/Qwen2.5-0.5B-Instruct")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "500"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
# Cosine similarity at or above which a new question reuses a cached answer.
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1800"))  # shorter prompt = faster CPU generation
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "0"))  # 0 = leave PyTorch default

# ---------------------------------------------------------------------------
# Usage / observability (all open source, nothing leaves your machine by default)
# ---------------------------------------------------------------------------
USAGE_DB_PATH = Path(os.getenv("USAGE_DB_PATH", VECTORSTORE_PATH.parent / "usage.db"))
# The models run locally, so real API cost is zero. These ASSUMED reference prices (USD per
# 1M tokens of a small hosted model) only estimate what the same traffic would cost on a paid API.
PRICE_PER_1M_INPUT_USD = float(os.getenv("PRICE_PER_1M_INPUT_USD", "0.15"))
PRICE_PER_1M_OUTPUT_USD = float(os.getenv("PRICE_PER_1M_OUTPUT_USD", "0.60"))
# Optional self-hosted Langfuse (open source). Tracing turns on only when all three are set.
# The keys come from YOUR OWN Langfuse instance, they are not a third-party API key.
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")