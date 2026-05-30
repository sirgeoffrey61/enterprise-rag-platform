"""In-memory request tracing for Enterprise RAG queries."""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class QueryTracer:
    def __init__(self, max_traces: int = 100) -> None:
        self.max_traces = max_traces
        self._traces: deque[dict[str, Any]] = deque(maxlen=max_traces)
        self._by_id: dict[str, dict[str, Any]] = {}

    def trace_query(self, query_id: str, query: str) -> None:
        trace = {
            "query_id": query_id,
            "query": query,
            "started_at": time.time(),
            "retrieval": None,
            "reranking": None,
            "generation": None,
        }
        self._by_id[query_id] = trace
        self._traces.append(trace)

    def record_retrieval(
        self, query_id: str, chunks_found: int, latency_ms: float
    ) -> None:
        trace = self._require_trace(query_id)
        trace["retrieval"] = {
            "chunks_found": chunks_found,
            "latency_ms": latency_ms,
        }

    def record_reranking(
        self, query_id: str, top_chunk_title: str, latency_ms: float
    ) -> None:
        trace = self._require_trace(query_id)
        trace["reranking"] = {
            "top_chunk_title": top_chunk_title,
            "latency_ms": latency_ms,
        }

    def record_generation(
        self,
        query_id: str,
        grounding_ratio: float,
        cache_hit: bool,
        latency_ms: float,
    ) -> None:
        trace = self._require_trace(query_id)
        trace["generation"] = {
            "grounding_ratio": grounding_ratio,
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
        }

    def get_trace(self, query_id: str) -> dict[str, Any] | None:
        return self._by_id.get(query_id)

    def last_traces(self, count: int = 5) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        return list(self._traces)[-count:]

    def _require_trace(self, query_id: str) -> dict[str, Any]:
        trace = self._by_id.get(query_id)
        if trace is None:
            raise KeyError(f"No trace found for query_id={query_id}")
        return trace
