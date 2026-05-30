"""Embed processed chunks for the Enterprise RAG retrieval pipeline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch  # after sentence_transformers (cu128 nightly import order)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 128

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "chunk_metadata.jsonl"

METADATA_FIELDS = ("chunk_id", "article_id", "title", "url", "text", "token_count")


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
        logger.warning(
            "CUDA is reported available but failed a device probe; falling back to CPU"
        )
        return "cpu"


def device_label(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def count_chunks(path: Path) -> int:
    with path.open(encoding="utf-8") as in_file:
        return sum(1 for _ in in_file)


def iter_chunk_batches(path: Path, batch_size: int) -> Iterator[tuple[list[str], list[dict]]]:
    texts: list[str] = []
    metadata: list[dict] = []
    with path.open(encoding="utf-8") as in_file:
        for line in in_file:
            chunk = json.loads(line)
            texts.append(chunk["text"])
            metadata.append({field: chunk[field] for field in METADATA_FIELDS})
            if len(texts) >= batch_size:
                yield texts, metadata
                texts, metadata = [], []
        if texts:
            yield texts, metadata


def embed_chunks_streaming(
    input_path: Path,
    *,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
) -> tuple[np.ndarray, int]:
    device = device or resolve_device()
    total = count_chunks(input_path)
    logger.info("Loading model {} on {}", model_name, device)
    model = SentenceTransformer(model_name, device=device)

    embedding_dim = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((total, embedding_dim), dtype=np.float32)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with METADATA_PATH.open("w", encoding="utf-8") as meta_file:
        for texts, metadata in tqdm(
            iter_chunk_batches(input_path, batch_size),
            desc="Embedding batches",
            unit="batch",
        ):
            batch_vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            batch_vectors = np.asarray(batch_vectors, dtype=np.float32)
            end = offset + len(texts)
            embeddings[offset:end] = batch_vectors
            offset = end

            for record in metadata:
                meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return embeddings, embedding_dim


def print_stats(
    *,
    total_chunks: int,
    embedding_dim: int,
    elapsed_seconds: float,
    device: str,
) -> None:
    print(f"Total chunks embedded: {total_chunks}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Time taken: {elapsed_seconds:.2f}s")
    print(f"Device used: {device_label(device)}")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()

    device = resolve_device()
    if device == "cuda":
        logger.info("Using GPU: {}", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA not available; using CPU")

    total = count_chunks(INPUT_PATH)
    logger.info("Processing {} chunks from {}", total, INPUT_PATH)

    start = time.perf_counter()
    embeddings, embedding_dim = embed_chunks_streaming(INPUT_PATH, device=device)
    elapsed = time.perf_counter() - start

    np.save(EMBEDDINGS_PATH, embeddings)
    logger.info("Saved embeddings to {}", EMBEDDINGS_PATH)
    logger.info("Saved metadata to {}", METADATA_PATH)

    print_stats(
        total_chunks=total,
        embedding_dim=embedding_dim,
        elapsed_seconds=elapsed,
        device=device,
    )


if __name__ == "__main__":
    main()
