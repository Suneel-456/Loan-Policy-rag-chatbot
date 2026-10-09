"""Hallucination (grounding) check: is the answer supported by the retrieved excerpts?

Two cheap, deterministic signals, no extra model and no extra latency:
  1. Every number in the answer (25,000 / 80% / 4% / 90) must also appear in the excerpts.
  2. Support score = share of the answer's content words found in the excerpts.
It catches invented figures and drifting answers. It does NOT catch a wrong conclusion built
from correct words; for that add an NLI model or an LLM judge (see README, "Next steps").
"""
import re
from dataclasses import dataclass, field

from src import config

_STOP = set(
    "that this with from have been will your their there which would should could about "
    "than then them they what when where while also into only does done more most other "
    "such these those each both some very must shall may can the and for are was were not "
    "you any all per its".split()
)
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_FAKE_PREFIX = re.compile(r"^\[FAKE LLM\]\s*")


@dataclass
class GroundingResult:
    grounded: bool
    score: float
    ungrounded_numbers: list[str] = field(default_factory=list)


def _norm_num(s: str) -> str:
    return s.replace(",", "").rstrip(".")


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]{4,}", text.lower()):
        if w not in _STOP:
            out.add(w[:-1] if w.endswith("s") else w)
    return out


def check_grounding(answer: str, context: str) -> GroundingResult:
    answer = _FAKE_PREFIX.sub("", answer).strip()
    if not answer or answer == config.NOT_FOUND_ANSWER:
        return GroundingResult(True, 1.0)

    ctx_numbers = {_norm_num(n) for n in _NUM.findall(context)}
    bad = [n for n in _NUM.findall(answer) if _norm_num(n) not in ctx_numbers]

    answer_words = _words(answer)
    score = 1.0 if not answer_words else len(answer_words & _words(context)) / len(answer_words)
    grounded = not bad and score >= config.GROUNDING_MIN_SCORE
    return GroundingResult(grounded, round(score, 3), bad)