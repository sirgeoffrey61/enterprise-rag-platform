"""Prometheus metrics and in-process aggregates for Enterprise RAG."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

QUERY_TOTAL = Counter(
    "query_total",
    "Total queries processed",
    labelnames=["cache_hit"],
)
QUERY_LATENCY_SECONDS = Histogram(
    "query_latency_seconds",
    "Full pipeline latency in seconds",
)
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "retrieval_latency_seconds",
    "Hybrid retrieval latency in seconds",
)
RERANK_LATENCY_SECONDS = Histogram(
    "rerank_latency_seconds",
    "Cross-encoder reranking latency in seconds",
)
GENERATION_LATENCY_SECONDS = Histogram(
    "generation_latency_seconds",
    "LLM generation latency in seconds",
)
GROUNDING_RATIO_HISTOGRAM = Histogram(
    "grounding_ratio_histogram",
    "Distribution of grounding ratios",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)
CACHE_HITS_TOTAL = Counter("cache_hits_total", "Total cache hits")
CACHE_MISSES_TOTAL = Counter("cache_misses_total", "Total cache misses")
INSUFFICIENT_EVIDENCE_TOTAL = Counter(
    "insufficient_evidence_total",
    "Total responses with insufficient evidence",
)


class MetricsRecorder:
    """Records Prometheus metrics and running aggregates for reports."""

    def __init__(self) -> None:
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.insufficient_evidence_count = 0
        self._latency_ms_sum = 0.0
        self._grounding_ratio_sum = 0.0

    def record_query(
        self,
        *,
        cache_hit: bool,
        query_latency_seconds: float,
        retrieval_latency_seconds: float,
        rerank_latency_seconds: float,
        generation_latency_seconds: float,
        grounding_ratio: float,
        insufficient_evidence: bool,
    ) -> None:
        cache_label = "true" if cache_hit else "false"
        QUERY_TOTAL.labels(cache_hit=cache_label).inc()
        QUERY_LATENCY_SECONDS.observe(query_latency_seconds)
        RETRIEVAL_LATENCY_SECONDS.observe(retrieval_latency_seconds)
        RERANK_LATENCY_SECONDS.observe(rerank_latency_seconds)
        GENERATION_LATENCY_SECONDS.observe(generation_latency_seconds)
        GROUNDING_RATIO_HISTOGRAM.observe(grounding_ratio)

        if cache_hit:
            CACHE_HITS_TOTAL.inc()
            self.cache_hits += 1
        else:
            CACHE_MISSES_TOTAL.inc()
            self.cache_misses += 1

        if insufficient_evidence:
            INSUFFICIENT_EVIDENCE_TOTAL.inc()
            self.insufficient_evidence_count += 1

        self.total_queries += 1
        self._latency_ms_sum += query_latency_seconds * 1000
        self._grounding_ratio_sum += grounding_ratio

    @property
    def cache_hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def avg_latency_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self._latency_ms_sum / self.total_queries

    @property
    def avg_grounding_ratio(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self._grounding_ratio_sum / self.total_queries

    @property
    def insufficient_evidence_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.insufficient_evidence_count / self.total_queries
