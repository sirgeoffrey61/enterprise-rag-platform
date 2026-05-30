"""Cross-encoder reranking for Enterprise RAG retrieval results."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import CrossEncoder

import torch  # must be imported after sentence_transformers on Windows/CUDA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError:
        logger.warning("CUDA probe failed; using CPU for cross-encoder")
        return "cpu"


class CrossEncoderReranker:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.device = device or resolve_device()
        logger.info("Loading cross-encoder {} on {}", model_name, self.device)
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []

        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)

        scored = []
        for chunk, score in zip(chunks, scores):
            row = copy.copy(chunk)
            row["rerank_score"] = float(score)
            row["rrf_score"] = chunk.get("rrf_score")
            scored.append(row)

        scored.sort(key=lambda row: row["rerank_score"], reverse=True)

        for rank, row in enumerate(scored[:top_k], start=1):
            row["rerank_rank"] = rank

        return scored[:top_k]

    def rerank_with_delta(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        rrf_ranks = {chunk["chunk_id"]: index + 1 for index, chunk in enumerate(chunks)}
        results = self.rerank(query, chunks, top_k=top_k)

        for row in results:
            rrf_rank = rrf_ranks[row["chunk_id"]]
            row["rrf_rank"] = rrf_rank
            row["rank_delta"] = rrf_rank - row["rerank_rank"]

        return results


def release_hybrid_gpu(retriever: object) -> None:
    """Drop the bi-encoder and free CUDA memory before loading the cross-encoder."""
    if hasattr(retriever, "model") and retriever.model is not None:
        del retriever.model
        retriever.model = None  # type: ignore[attr-defined]
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Released hybrid retriever GPU memory")


def run_tests() -> None:
    from retrieval.hybrid_retriever import HybridRetriever

    queries = [
        "What is anarchism political philosophy?",
        "History of the Roman Empire",
        "Machine learning neural networks",
    ]

    retriever = HybridRetriever()
    candidates_by_query: dict[str, list[dict]] = {}

    logger.info("Running hybrid retrieval for {} queries", len(queries))
    for query in queries:
        candidates_by_query[query] = retriever.search(query, top_k=25, candidate_k=25)

    release_hybrid_gpu(retriever)

    reranker = CrossEncoderReranker(device="cpu")

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 72)

        candidates = candidates_by_query[query]
        rrf_top_title = candidates[0]["title"] if candidates else "N/A"

        results = reranker.rerank_with_delta(query, candidates, top_k=5)
        rerank_top_title = results[0]["title"] if results else "N/A"

        order_changed = sum(
            1
            for row in results
            if row["rank_delta"] != 0
        )
        top_changed = rrf_top_title != rerank_top_title

        print(f"RRF #1: {rrf_top_title}  ->  Rerank #1: {rerank_top_title}")
        print(
            f"Order impact: {order_changed}/{len(results)} results moved rank; "
            f"top result {'CHANGED' if top_changed else 'unchanged'}"
        )
        print()

        for row in results:
            print(
                f"  {row['rerank_rank']}. {row['title']} | "
                f"rrf={row['rrf_score']:.6f} rerank={row['rerank_score']:.4f} "
                f"delta={row['rank_delta']:+d} (rrf_rank={row['rrf_rank']})"
            )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()
    run_tests()


if __name__ == "__main__":
    main()
