"""Model routing: use the cheapest model that can answer, escalate only when needed.

simple   -> small model (Qwen2.5-0.5B): short lookup questions with no numbers
complex  -> main model  (LLM_MODEL)   : comparisons, "why/explain", multi-part questions,
                                        and anything containing a number (amount or months to apply
                                        a rule to, which the small model often gets wrong)
Cascade: if the small model's answer fails the grounding check, the bot re-asks the main model.
With ROUTING_ENABLED=0 (default) every question goes to the main model.
The small model is loaded lazily on first use so it costs no memory until needed.
"""
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src import config

_COMPLEX_HINTS = re.compile(
    r"\b(compare|comparison|difference|versus|vs|why|explain|calculate|what if|both|"
    r"scenario|step by step|pros and cons|eligible if|and also|summari[sz]e)\b",
    re.I,
)


def classify(question: str) -> str:
    """Return "simple" or "complex" using cheap rules (no model call)."""
    if len(question.split()) > 25:
        return "complex"
    if question.count("?") > 1:
        return "complex"
    if re.search(r"\d", question):
        return "complex"
    if _COMPLEX_HINTS.search(question):
        return "complex"
    return "simple"


@dataclass
class Route:
    tier: str      # "simple" or "complex"
    llm: Any
    model: str


class ModelRouter:
    def __init__(self, main_llm, small_factory: Optional[Callable[[], Any]] = None, enabled: bool = None):
        self.main = main_llm
        self.small_factory = small_factory
        self.enabled = config.ROUTING_ENABLED if enabled is None else enabled
        self._small = None
        self._lock = threading.Lock()

    @staticmethod
    def _name(llm) -> str:
        return getattr(llm, "model_id", llm.__class__.__name__)

    def _small_llm(self):
        with self._lock:
            if self._small is None:
                self._small = self.small_factory()
            return self._small

    def route(self, question: str) -> Route:
        tier = classify(question)
        if self.enabled and tier == "simple" and self.small_factory is not None:
            llm = self._small_llm()
            return Route("simple", llm, self._name(llm))
        return Route("complex", self.main, self._name(self.main))

    def main_route(self) -> Route:
        return Route("complex", self.main, self._name(self.main))