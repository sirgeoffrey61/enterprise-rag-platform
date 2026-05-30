"""Hybrid retrieval: BM25 sparse + Qdrant dense with RRF fusion."""

from __future__ import annotations

import json
import pickle
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import torch  # must be imported after sentence_transformers on Windows/CUDA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "chunk_metadata.jsonl"
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "bm25_index.pkl"

COLLECTION_NAME = "enterprise_rag"
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
RRF_DENSE_WEIGHT = 0.5
RRF_BM25_WEIGHT = 0.5


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
        logger.warning("CUDA probe failed; using CPU for query embedding")
        return "cpu"


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def load_metadata(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as in_file:
        for line in in_file:
            records.append(json.loads(line))
    return records


class HybridRetriever:
    def __init__(
        self,
        *,
        metadata_path: Path = METADATA_PATH,
        bm25_index_path: Path = BM25_INDEX_PATH,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.bm25_index_path = bm25_index_path
        self.collection_name = collection_name
        self.device = device or resolve_device()

        self.records = load_metadata(metadata_path)
        self._chunk_by_id = {record["chunk_id"]: record for record in self.records}

        self.bm25, self._bm25_corpus_ids = self._load_or_build_bm25()
        self.qdrant = QdrantClient(url=qdrant_url, check_compatibility=False)

        logger.info("Loading model {} on {}", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def _load_or_build_bm25(self) -> tuple[BM25Okapi, list[str]]:
        if self.bm25_index_path.exists():
            logger.info("Loading BM25 index from {}", self.bm25_index_path)
            with self.bm25_index_path.open("rb") as in_file:
                saved = pickle.load(in_file)
            if saved.get("metadata_path") != str(self.metadata_path):
                logger.warning("Metadata path changed; rebuilding BM25 index")
            elif len(saved["chunk_ids"]) != len(self.records):
                logger.warning("Chunk count changed; rebuilding BM25 index")
            else:
                return saved["bm25"], saved["chunk_ids"]

        logger.info("Building BM25 index from {} chunks", len(self.records))
        tokenized_corpus = [tokenize(record["text"]) for record in self.records]
        chunk_ids = [record["chunk_id"] for record in self.records]
        bm25 = BM25Okapi(tokenized_corpus)

        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.bm25_index_path.open("wb") as out_file:
            pickle.dump(
                {
                    "bm25": bm25,
                    "chunk_ids": chunk_ids,
                    "metadata_path": str(self.metadata_path),
                },
                out_file,
            )
        logger.info("Saved BM25 index to {}", self.bm25_index_path)
        return bm25, chunk_ids

    def embed_query(self, query: str) -> list[float]:
        prefixed = f"{QUERY_PREFIX}{query}"
        vector = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def dense_search(self, query: str, candidate_k: int) -> list[dict]:
        vector = self.embed_query(query)
        hits = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=candidate_k,
            with_payload=True,
        ).points

        results: list[dict] = []
        for rank, point in enumerate(hits, start=1):
            payload = point.payload or {}
            chunk_id = payload["chunk_id"]
            record = self._chunk_by_id.get(chunk_id, payload)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "title": record["title"],
                    "url": record["url"],
                    "text": record["text"],
                    "token_count": record["token_count"],
                    "dense_rank": rank,
                }
            )
        return results

    def bm25_search(self, query: str, candidate_k: int) -> list[dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:candidate_k]

        results: list[dict] = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk_id = self._bm25_corpus_ids[index]
            record = self._chunk_by_id[chunk_id]
            results.append(
                {
                    "chunk_id": chunk_id,
                    "title": record["title"],
                    "url": record["url"],
                    "text": record["text"],
                    "token_count": record["token_count"],
                    "bm25_rank": rank,
                }
            )
        return results

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[dict],
        bm25_results: list[dict],
        *,
        rrf_k: int = RRF_K,
        dense_weight: float = RRF_DENSE_WEIGHT,
        bm25_weight: float = RRF_BM25_WEIGHT,
    ) -> list[dict]:
        fused_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}
        records: dict[str, dict] = {}

        for item in dense_results:
            chunk_id = item["chunk_id"]
            records[chunk_id] = item
            dense_ranks[chunk_id] = item["dense_rank"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + dense_weight / (
                rrf_k + item["dense_rank"]
            )

        for item in bm25_results:
            chunk_id = item["chunk_id"]
            records.setdefault(chunk_id, item)
            bm25_ranks[chunk_id] = item["bm25_rank"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + bm25_weight / (
                rrf_k + item["bm25_rank"]
            )

        fused: list[dict] = []
        for chunk_id, rrf_score in fused_scores.items():
            base = records[chunk_id]
            fused.append(
                {
                    "chunk_id": chunk_id,
                    "title": base["title"],
                    "url": base["url"],
                    "text": base["text"],
                    "token_count": base["token_count"],
                    "dense_rank": dense_ranks.get(chunk_id),
                    "bm25_rank": bm25_ranks.get(chunk_id),
                    "rrf_score": rrf_score,
                }
            )

        fused.sort(key=lambda row: row["rrf_score"], reverse=True)
        return fused

    def search(self, query: str, top_k: int = 5, candidate_k: int = 25) -> list[dict]:
        dense_results = self.dense_search(query, candidate_k)
        bm25_results = self.bm25_search(query, candidate_k)
        fused = self.reciprocal_rank_fusion(dense_results, bm25_results)
        return fused[:top_k]


def run_tests(retriever: HybridRetriever) -> None:
    queries = [
        "What is anarchism political philosophy?",
        "History of the Roman Empire",
        "Machine learning neural networks",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retriever.search(query, top_k=3, candidate_k=25)
        for index, result in enumerate(results, start=1):
            print(
                f"{index}. {result['title']} | rrf_score={result['rrf_score']:.6f} "
                f"(dense={result['dense_rank']}, bm25={result['bm25_rank']})"
            )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()
    retriever = HybridRetriever()
    run_tests(retriever)


if __name__ == "__main__":
    main()
