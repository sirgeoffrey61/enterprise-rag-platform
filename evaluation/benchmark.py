"""Continuous benchmark runner for Enterprise RAG pipeline."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.cache import QueryCache
from api.generator import INSUFFICIENT_EVIDENCE, GroundedGenerator, answer_query
from reranking.reranker import CrossEncoderReranker, release_hybrid_gpu
from retrieval.hybrid_retriever import HybridRetriever

BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "benchmark_queries.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "benchmark_results.json"

# Estimated rough blended token cost per query (USD) for reporting only.
ESTIMATED_COST_PER_QUERY_USD = 0.0004


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class EvaluationRunner:
    def __init__(self, benchmark_path: Path = BENCHMARK_PATH) -> None:
        self.benchmark_path = benchmark_path
        self.cache = QueryCache()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(device="cpu")
        self.generator = GroundedGenerator()

    def _load_queries(self) -> list[dict]:
        with self.benchmark_path.open(encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _is_insufficient(answer: str) -> bool:
        return INSUFFICIENT_EVIDENCE in answer

    def _answer_with_retries(self, query: str, *, max_retries: int = 3) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return answer_query(
                    query,
                    cache=self.cache,
                    retriever=self.retriever,
                    reranker=self.reranker,
                    generator=self.generator,
                    release_gpu_after=False,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Query failed (attempt {}/{}): {} | {}",
                    attempt,
                    max_retries,
                    query,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        raise last_error  # type: ignore[misc]

    def run_benchmark(self, *, release_gpu_after: bool = True) -> dict:
        rows = self._load_queries()
        results: list[dict] = []

        for idx, row in enumerate(rows, start=1):
            query = row["query"]
            logger.info("Benchmark {}/{}: {}", idx, len(rows), query)

            started = time.perf_counter()
            try:
                output = self._answer_with_retries(query)
                elapsed_ms = (time.perf_counter() - started) * 1000
                insufficient = self._is_insufficient(output["answer"])
                answered = not insufficient
                timing = output.get("timings", {})
                total_ms = float(timing.get("total_ms", elapsed_ms))
                error = None
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                insufficient = True
                answered = False
                total_ms = elapsed_ms
                output = {"grounding_ratio": 0.0, "cache_hit": False, "cited_chunk_ids": []}
                error = str(exc)

            results.append(
                {
                    "query": query,
                    "expected_type": row["expected_type"],
                    "topic": row["topic"],
                    "answered": answered,
                    "insufficient_evidence": insufficient,
                    "grounding_ratio": output.get("grounding_ratio", 0.0),
                    "latency_ms": total_ms,
                    "cache_hit": output.get("cache_hit", False),
                    "cited_chunk_ids_count": len(output.get("cited_chunk_ids", [])),
                    "error": error,
                }
            )

        if release_gpu_after and getattr(self.retriever, "model", None) is not None:
            release_hybrid_gpu(self.retriever)

        metrics = self._compute_metrics(results)
        report = {
            "generated_at": time.time(),
            "total_queries": len(results),
            "metrics": metrics,
            "results": results,
        }
        self._save_results(report)
        return report

    @staticmethod
    def _compute_metrics(results: list[dict]) -> dict:
        answerable = [r for r in results if r["expected_type"] == "answerable"]
        unanswerable = [r for r in results if r["expected_type"] == "unanswerable"]

        recall_hits = sum(1 for r in answerable if r["answered"])
        precision_hits = sum(1 for r in unanswerable if r["insufficient_evidence"])

        answerable_grounding = [r["grounding_ratio"] for r in answerable]
        miss_latencies = [r["latency_ms"] for r in results if not r["cache_hit"]]
        insufficient_count = sum(1 for r in results if r["insufficient_evidence"])

        return {
            "recall_answerable_hits": recall_hits,
            "recall_answerable_total": len(answerable),
            "precision_unanswerable_hits": precision_hits,
            "precision_unanswerable_total": len(unanswerable),
            "avg_grounding_ratio_answerable": (
                statistics.mean(answerable_grounding) if answerable_grounding else 0.0
            ),
            "avg_latency_ms_no_cache": statistics.mean(miss_latencies) if miss_latencies else 0.0,
            "insufficient_evidence_rate": insufficient_count / len(results) if results else 0.0,
            "estimated_total_cost_usd": len(results) * ESTIMATED_COST_PER_QUERY_USD,
        }

    def _save_results(self, report: dict) -> None:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_PATH.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Saved benchmark results to {}", RESULTS_PATH)


def print_report(report: dict) -> None:
    m = report["metrics"]
    total = report["total_queries"]
    recall = f"{m['recall_answerable_hits']}/{m['recall_answerable_total']}"
    precision = f"{m['precision_unanswerable_hits']}/{m['precision_unanswerable_total']}"

    print("=====================================")
    print("BENCHMARK REPORT")
    print("=====================================")
    print(f"Total queries: {total}")
    print(f"Recall (answerable): {recall}")
    print(f"Precision (unanswerable): {precision}")
    print(f"Avg grounding ratio: {m['avg_grounding_ratio_answerable']:.2f}")
    print(f"Avg latency (no cache): {m['avg_latency_ms_no_cache']:.0f}ms")
    print(f"Insufficient evidence rate: {m['insufficient_evidence_rate'] * 100:.1f}%")
    print(f"Estimated total cost: ${m['estimated_total_cost_usd']:.4f}")
    print("=====================================")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise ValueError("GROQ_API_KEY is required in .env to run benchmark.")
    runner = EvaluationRunner()
    report = runner.run_benchmark()
    print_report(report)


if __name__ == "__main__":
    main()
