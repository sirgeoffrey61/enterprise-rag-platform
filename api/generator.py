"""Grounded answer generation with inline citations via Groq."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from observability.metrics import MetricsRecorder
from observability.tracer import QueryTracer

MODEL_NAME = "llama-3.1-8b-instant"
MAX_TOKENS = 1024
INSUFFICIENT_EVIDENCE = (
    "INSUFFICIENT EVIDENCE: I cannot answer this from the provided context."
)
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """You are a grounded research assistant for an Enterprise RAG system.

Rules:
- Answer ONLY using the numbered context passages below.
- Cite sources inline using [1], [2], [3], etc. matching the passage numbers.
- Do NOT use outside knowledge, guesses, or assumptions.
- If the context does not contain enough information to answer, respond with exactly:
  INSUFFICIENT EVIDENCE: I cannot answer this from the provided context.
"""

_metrics = MetricsRecorder()
_tracer = QueryTracer()


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def get_metrics_recorder() -> MetricsRecorder:
    return _metrics


def get_tracer() -> QueryTracer:
    return _tracer


def get_observability_report() -> dict:
    return {
        "total_queries": _metrics.total_queries,
        "cache_hit_rate": round(_metrics.cache_hit_rate, 4),
        "avg_latency_ms": round(_metrics.avg_latency_ms, 2),
        "avg_grounding_ratio": round(_metrics.avg_grounding_ratio, 4),
        "insufficient_evidence_rate": round(_metrics.insufficient_evidence_rate, 4),
        "last_5_traces": _tracer.last_traces(5),
    }


def build_context_block(chunks: list[dict]) -> str:
    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(f"[{index}] Title: {chunk['title']}\n{chunk['text']}\n---")
    return "\n".join(sections)


def is_insufficient_evidence(answer: str) -> bool:
    return INSUFFICIENT_EVIDENCE in answer


class GroundedGenerator:
    def __init__(self, api_key: str | None = None, model: str = MODEL_NAME) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to .env before using GroundedGenerator."
            )
        self.model = model
        self.max_tokens = MAX_TOKENS
        self.client = Groq(api_key=key)
        logger.info("GroundedGenerator ready (model={})", self.model)

    def generate(
        self,
        query: str,
        reranked_chunks: list[dict],
        max_chunks: int = 5,
    ) -> dict:
        chunks = reranked_chunks[:max_chunks]
        context = build_context_block(chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content or ""

        validation = self.validate_citations(answer, chunks)
        cited_chunk_ids: list[str] = []
        seen_ids: set[str] = set()
        for index in validation["valid_citations"]:
            chunk_id = chunks[index - 1]["chunk_id"]
            if chunk_id not in seen_ids:
                cited_chunk_ids.append(chunk_id)
                seen_ids.add(chunk_id)

        return {
            "answer": answer,
            "cited_chunk_ids": cited_chunk_ids,
            "grounding_ratio": validation["grounding_ratio"],
            "raw_chunks": chunks,
            "validation": validation,
        }

    def validate_citations(self, answer: str, chunks: list[dict]) -> dict:
        cited_numbers = [int(match) for match in CITATION_PATTERN.findall(answer)]
        valid_numbers: list[int] = []
        invalid_numbers: list[int] = []
        seen_valid: set[int] = set()

        for number in cited_numbers:
            if 1 <= number <= len(chunks):
                valid_numbers.append(number)
                seen_valid.add(number)
            else:
                invalid_numbers.append(number)

        total_chunks = len(chunks)
        grounding_ratio = len(seen_valid) / total_chunks if total_chunks else 0.0

        return {
            "valid_citations": valid_numbers,
            "invalid_citations": invalid_numbers,
            "grounding_ratio": grounding_ratio,
        }


def run_full_pipeline(
    query: str,
    *,
    query_id: str | None = None,
    candidate_k: int = 25,
    rerank_top_k: int = 5,
    max_chunks: int = 5,
    retriever: object | None = None,
    reranker: object | None = None,
    generator: GroundedGenerator | None = None,
    own_retriever: bool = False,
    release_gpu_after: bool = False,
    metrics: MetricsRecorder | None = None,
    tracer: QueryTracer | None = None,
    skip_trace: bool = False,
) -> dict:
    """Hybrid search -> rerank -> grounded generation for one query."""
    from reranking.reranker import CrossEncoderReranker, release_hybrid_gpu
    from retrieval.hybrid_retriever import HybridRetriever

    metrics = metrics or _metrics
    tracer = tracer or _tracer
    query_id = query_id or str(uuid.uuid4())
    if not skip_trace:
        tracer.trace_query(query_id, query)

    pipeline_start = time.perf_counter()

    if retriever is None:
        retriever = HybridRetriever()
        own_retriever = True

    retrieval_start = time.perf_counter()
    candidates = retriever.search(query, top_k=candidate_k, candidate_k=candidate_k)
    retrieval_seconds = time.perf_counter() - retrieval_start
    tracer.record_retrieval(
        query_id, len(candidates), retrieval_seconds * 1000
    )

    if own_retriever or release_gpu_after:
        release_hybrid_gpu(retriever)
        retriever = None

    if reranker is None:
        reranker = CrossEncoderReranker(device="cpu")

    rerank_start = time.perf_counter()
    reranked = reranker.rerank(query, candidates, top_k=rerank_top_k)
    rerank_seconds = time.perf_counter() - rerank_start
    top_title = reranked[0]["title"] if reranked else ""
    tracer.record_reranking(query_id, top_title, rerank_seconds * 1000)

    if generator is None:
        generator = GroundedGenerator()

    generation_start = time.perf_counter()
    result = generator.generate(query, reranked, max_chunks=max_chunks)
    generation_seconds = time.perf_counter() - generation_start

    total_seconds = time.perf_counter() - pipeline_start
    serialized = _serialize_pipeline_result(result, query=query, query_id=query_id)
    insufficient = is_insufficient_evidence(serialized["answer"])

    tracer.record_generation(
        query_id,
        serialized["grounding_ratio"],
        cache_hit=False,
        latency_ms=generation_seconds * 1000,
    )
    metrics.record_query(
        cache_hit=False,
        query_latency_seconds=total_seconds,
        retrieval_latency_seconds=retrieval_seconds,
        rerank_latency_seconds=rerank_seconds,
        generation_latency_seconds=generation_seconds,
        grounding_ratio=serialized["grounding_ratio"],
        insufficient_evidence=insufficient,
    )

    serialized["timings"] = {
        "retrieval_ms": retrieval_seconds * 1000,
        "rerank_ms": rerank_seconds * 1000,
        "generation_ms": generation_seconds * 1000,
        "total_ms": total_seconds * 1000,
    }
    return serialized


def _serialize_pipeline_result(
    result: dict,
    *,
    query: str | None = None,
    query_id: str | None = None,
) -> dict:
    """JSON-safe payload for Redis (omit non-serializable validation details)."""
    return {
        "query": query or result.get("query"),
        "query_id": query_id,
        "answer": result["answer"],
        "cited_chunk_ids": result["cited_chunk_ids"],
        "grounding_ratio": result["grounding_ratio"],
        "raw_chunks": result["raw_chunks"],
    }


def answer_query(
    query: str,
    cache: object | None = None,
    *,
    query_id: str | None = None,
    candidate_k: int = 25,
    rerank_top_k: int = 5,
    max_chunks: int = 5,
    retriever: object | None = None,
    reranker: object | None = None,
    generator: GroundedGenerator | None = None,
    release_gpu_after: bool = False,
    metrics: MetricsRecorder | None = None,
    tracer: QueryTracer | None = None,
) -> dict:
    """Answer with Redis cache: return cached result or run full pipeline."""
    from api.cache import QueryCache

    metrics = metrics or _metrics
    tracer = tracer or _tracer
    query_id = query_id or str(uuid.uuid4())
    tracer.trace_query(query_id, query)

    cache = cache or QueryCache()
    pipeline_start = time.perf_counter()

    cached = cache.get(query)
    if cached is not None:
        total_seconds = time.perf_counter() - pipeline_start
        result = dict(cached)
        result["cache_hit"] = True
        result["query_id"] = query_id
        insufficient = is_insufficient_evidence(result["answer"])
        chunks_found = len(result.get("raw_chunks", []))

        tracer.record_retrieval(query_id, chunks_found, 0.0)
        tracer.record_reranking(query_id, "", 0.0)
        tracer.record_generation(
            query_id,
            result["grounding_ratio"],
            cache_hit=True,
            latency_ms=total_seconds * 1000,
        )
        metrics.record_query(
            cache_hit=True,
            query_latency_seconds=total_seconds,
            retrieval_latency_seconds=0.0,
            rerank_latency_seconds=0.0,
            generation_latency_seconds=0.0,
            grounding_ratio=result["grounding_ratio"],
            insufficient_evidence=insufficient,
        )
        result["timings"] = {
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": total_seconds * 1000,
        }
        return result

    result = run_full_pipeline(
        query,
        query_id=query_id,
        candidate_k=candidate_k,
        rerank_top_k=rerank_top_k,
        max_chunks=max_chunks,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        own_retriever=retriever is None,
        release_gpu_after=release_gpu_after,
        metrics=metrics,
        tracer=tracer,
        skip_trace=True,
    )
    result["cache_hit"] = False
    cache.set(query, {k: v for k, v in result.items() if k != "timings"})
    return result


def run_pipeline_tests() -> None:
    from retrieval.hybrid_retriever import HybridRetriever
    from reranking.reranker import CrossEncoderReranker, release_hybrid_gpu

    queries = [
        "What is anarchism political philosophy?",
        "What caused the fall of the Roman Empire?",
    ]

    retriever = HybridRetriever()
    hybrid_results: dict[str, list[dict]] = {}
    for query in queries:
        logger.info("Hybrid search: {}", query)
        hybrid_results[query] = retriever.search(query, top_k=25, candidate_k=25)

    release_hybrid_gpu(retriever)

    reranker = CrossEncoderReranker(device="cpu")
    reranked_results: dict[str, list[dict]] = {}
    for query in queries:
        logger.info("Reranking: {}", query)
        reranked_results[query] = reranker.rerank(query, hybrid_results[query], top_k=5)

    if not os.environ.get("GROQ_API_KEY", "").strip():
        print(
            "\nGROQ_API_KEY is empty in .env — hybrid retrieval and reranking completed.\n"
            "Add your key and re-run: .\\venv\\Scripts\\python.exe api\\generator.py\n"
        )
        return

    generator = GroundedGenerator()

    for query in queries:
        print(f"\nQuery: {query}")
        print("=" * 72)
        result = generator.generate(query, reranked_results[query], max_chunks=5)
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"grounding_ratio: {result['grounding_ratio']:.2f}")
        print(f"cited_chunk_ids: {result['cited_chunk_ids']}")


def run_cache_test() -> None:
    from api.cache import QueryCache

    query = "What is anarchism political philosophy?"
    cache = QueryCache()

    print(f"Cache keys before test: {cache.get_stats()['total_keys']}")
    print(f"\nQuery: {query}\n")

    start = time.perf_counter()
    first = answer_query(query, cache=cache)
    first_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    second = answer_query(query, cache=cache)
    second_ms = (time.perf_counter() - start) * 1000

    print("--- Run 1 (cache miss expected) ---")
    print(f"cache_hit: {first['cache_hit']}")
    print(f"latency_ms: {first_ms:.1f}")
    print(f"grounding_ratio: {first['grounding_ratio']:.2f}")
    print(f"cited_chunk_ids: {first['cited_chunk_ids']}")
    print(f"answer preview: {first['answer'][:200]}...")

    print("\n--- Run 2 (cache hit expected) ---")
    print(f"cache_hit: {second['cache_hit']}")
    print(f"latency_ms: {second_ms:.1f}")
    print(f"grounding_ratio: {second['grounding_ratio']:.2f}")
    print(f"cited_chunk_ids: {second['cited_chunk_ids']}")

    speedup = first_ms / second_ms if second_ms > 0 else float("inf")
    print(f"\nLatency: {first_ms:.1f} ms -> {second_ms:.1f} ms ({speedup:.1f}x faster)")
    print(f"Cache keys after test: {cache.get_stats()['total_keys']}")


def run_observability_test() -> None:
    from api.cache import QueryCache
    from reranking.reranker import CrossEncoderReranker, release_hybrid_gpu
    from retrieval.hybrid_retriever import HybridRetriever

    queries = [
        "What is anarchism political philosophy?",
        "What caused the fall of the Roman Empire?",
        "Machine learning neural networks",
        "History of ancient Egypt civilization",
        "Who was Albert Einstein?",
    ]

    cache = QueryCache()
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker(device="cpu")
    generator = GroundedGenerator()

    for index, query in enumerate(queries, start=1):
        logger.info("Observability test query {}/{}: {}", index, len(queries), query)
        answer_query(
            query,
            cache=cache,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            release_gpu_after=(index == 1),
        )

    if retriever is not None and getattr(retriever, "model", None) is not None:
        release_hybrid_gpu(retriever)

    report = get_observability_report()
    print("\n" + "=" * 72)
    print("OBSERVABILITY REPORT")
    print("=" * 72)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()

    if len(sys.argv) > 1 and sys.argv[1] == "--cache-test":
        if not os.environ.get("GROQ_API_KEY", "").strip():
            print("GROQ_API_KEY is required for cache test (run 1 calls Groq).")
            return
        run_cache_test()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--observability-test":
        if not os.environ.get("GROQ_API_KEY", "").strip():
            print("GROQ_API_KEY is required for observability test.")
            return
        run_observability_test()
        return

    run_pipeline_tests()


if __name__ == "__main__":
    main()
