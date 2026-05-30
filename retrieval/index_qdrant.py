"""Index chunk embeddings into Qdrant for the Enterprise RAG retrieval pipeline."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

COLLECTION_NAME = "enterprise_rag"
VECTOR_SIZE = 384
UPLOAD_BATCH_SIZE = 256
QDRANT_URL = "http://localhost:6333"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "chunk_metadata.jsonl"


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def load_metadata(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as in_file:
        for line in in_file:
            records.append(json.loads(line))
    return records


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def index_embeddings(
    *,
    embeddings_path: Path = EMBEDDINGS_PATH,
    metadata_path: Path = METADATA_PATH,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = UPLOAD_BATCH_SIZE,
) -> dict[str, str | int | float]:
    embeddings = np.load(embeddings_path)
    metadata = load_metadata(metadata_path)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match metadata ({len(metadata)})"
        )
    if embeddings.shape[1] != VECTOR_SIZE:
        raise ValueError(f"Expected vector size {VECTOR_SIZE}, got {embeddings.shape[1]}")

    client = QdrantClient(url=qdrant_url, check_compatibility=False)
    logger.info("Connected to Qdrant at {}", qdrant_url)

    if client.collection_exists(collection_name):
        logger.info("Recreating existing collection {}", collection_name)
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
            on_disk=True,
        ),
    )
    logger.info("Created collection {} (cosine, on_disk=True)", collection_name)

    total = len(metadata)
    start = time.perf_counter()

    for batch_start in tqdm(range(0, total, batch_size), desc="Uploading to Qdrant", unit="batch"):
        batch_end = min(batch_start + batch_size, total)
        points = [
            PointStruct(
                id=chunk_id_to_point_id(record["chunk_id"]),
                vector=embeddings[batch_start + offset].tolist(),
                payload=record,
            )
            for offset, record in enumerate(metadata[batch_start:batch_end])
        ]
        client.upsert(collection_name=collection_name, points=points)

    elapsed = time.perf_counter() - start
    collection_info = client.get_collection(collection_name)
    indexed_count = collection_info.points_count

    return {
        "collection_name": collection_name,
        "total_vectors_indexed": indexed_count,
        "time_taken": elapsed,
    }


def print_stats(stats: dict[str, str | int | float]) -> None:
    print(f"Collection name: {stats['collection_name']}")
    print(f"Total vectors indexed: {stats['total_vectors_indexed']}")
    print(f"Time taken: {stats['time_taken']:.2f}s")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()
    logger.info("Loading embeddings from {}", EMBEDDINGS_PATH)
    stats = index_embeddings()
    print_stats(stats)


if __name__ == "__main__":
    main()
