"""Input and output guardrails (rule based, no model, no extra latency).

Input : blocks prompt-injection attempts and requests to help with fraud.
Output: stops the system prompt from leaking and masks any PII in the answer.
Rules are intentionally simple and easy to extend. For heavier needs use NeMo Guardrails
or Llama Guard (both open source) in front of this.
"""
import re
from dataclasses import dataclass

from src.pii import mask_pii

BLOCK_MESSAGE = (
    "I can't help with that request. I can answer questions about the loan policy documents."
)

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your) (instructions|rules|prompts?)",
    r"(reveal|show|print|repeat|tell me) (me )?(your|the) (system |hidden |initial )?(prompt|instructions)",
    r"you are now\b",
    r"\bact as (an? )?(unrestricted|jailbroken|dan)\b",
    r"\b(jailbreak|developer mode|do anything now)\b",
    r"pretend (that )?you (have no|are not bound)",
]
_FRAUD = [
    r"\b(fake|forged|forge|fabricat\w*|doctored)\b.{0,40}\b(documents?|payslips?|salary slips?|statements?|itr|kyc)\b",
    r"\b(bypass|avoid|skip|evade)\b.{0,30}\b(kyc|credit check|verification|credit score)\b",
    r"\blaunder\w*\b",
    r"\bwithout (any )?(kyc|documents)\b.{0,30}\b(approve|approval|sanction)\b",
]
_INJECTION_RE = [re.compile(p, re.I) for p in _INJECTION]
_FRAUD_RE = [re.compile(p, re.I) for p in _FRAUD]
_LEAK_RE = re.compile(r"you are a loan policy assistant|answer the question using only the policy excerpts", re.I)


@dataclass
class GuardResult:
    allowed: bool
    category: str = ""   # "prompt_injection", "fraud", "prompt_leak" or ""
    message: str = ""


def check_input(question: str) -> GuardResult:
    for rx in _INJECTION_RE:
        if rx.search(question):
            return GuardResult(False, "prompt_injection", BLOCK_MESSAGE)
    for rx in _FRAUD_RE:
        if rx.search(question):
            return GuardResult(False, "fraud", BLOCK_MESSAGE)
    return GuardResult(True)


def check_output(answer: str) -> tuple[str, GuardResult, dict[str, int]]:
    """Return (safe_answer, result, pii_found_in_answer)."""
    if _LEAK_RE.search(answer):
        return BLOCK_MESSAGE, GuardResult(False, "prompt_leak", BLOCK_MESSAGE), {}
    masked, found = mask_pii(answer, use_presidio=False)
    return masked, GuardResult(True), found