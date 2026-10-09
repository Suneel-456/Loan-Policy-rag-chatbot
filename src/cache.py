"""Two-level answer cache: the biggest cost and latency saver for repeated questions.

Level 1 (exact)   : normalised question text -> answer (microseconds).
Level 2 (semantic): cosine similarity of question embeddings >= SEMANTIC_CACHE_THRESHOLD, so
                    "what's the min income for a personal loan" can reuse an earlier answer.
Only grounded, non-blocked answers are cached. The cache is in memory and cleared on /reindex.
"""
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from src import config


def normalize(question: str) -> str:
    return re.sub(r"[^a-z0-9% ]+", "", " ".join(question.lower().split()))


class AnswerCache:
    def __init__(self, size: int = None, ttl: int = None, threshold: float = None):
        self.size = config.CACHE_SIZE if size is None else size
        self.ttl = config.CACHE_TTL_SECONDS if ttl is None else ttl
        self.threshold = config.SEMANTIC_CACHE_THRESHOLD if threshold is None else threshold
        self._items: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def _alive(self, item: dict) -> bool:
        return time.time() - item["ts"] <= self.ttl

    def get(self, question: str, vector: Optional[list[float]] = None) -> Optional[tuple[Any, str]]:
        """Return (value, kind) where kind is "exact" or "semantic", or None."""
        key = normalize(question)
        with self._lock:
            item = self._items.get(key)
            if item and self._alive(item):
                self._items.move_to_end(key)
                return item["value"], "exact"
            if vector is not None:
                q = np.asarray(vector, dtype=float)
                qn = np.linalg.norm(q)
                best, best_sim = None, self.threshold
                for k, it in self._items.items():
                    if it["vector"] is None or not self._alive(it):
                        continue
                    v = it["vector"]
                    sim = float(np.dot(q, v) / (qn * np.linalg.norm(v) + 1e-12))
                    if sim >= best_sim:
                        best, best_sim = k, sim
                if best is not None:
                    self._items.move_to_end(best)
                    return self._items[best]["value"], "semantic"
        return None

    def put(self, question: str, value: Any, vector: Optional[list[float]] = None) -> None:
        with self._lock:
            self._items[normalize(question)] = {
                "value": value,
                "vector": None if vector is None else np.asarray(vector, dtype=float),
                "ts": time.time(),
            }
            self._items.move_to_end(normalize(question))
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)