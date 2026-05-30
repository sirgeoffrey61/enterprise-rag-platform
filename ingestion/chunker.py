"""Clean and chunk Wikipedia articles for the Enterprise RAG ingestion pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 100
ENCODING_NAME = "cl100k_base"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "wikipedia_raw.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

# Wikipedia markup patterns
_HEADING_RE = re.compile(r"={2,}\s*(.*?)\s*={2,}", re.MULTILINE)
_LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
_EXTERNAL_LINK_RE = re.compile(r"\[(?:https?://|//)[^\]]+\]", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\}", re.DOTALL)
_PUNCT_ONLY_LINE_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def remove_templates(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _TEMPLATE_RE.sub("", text)
    return text


def clean_text(raw: str) -> str:
    text = remove_templates(raw)
    text = _HEADING_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _EXTERNAL_LINK_RE.sub("", text)
    text = _HTML_RE.sub("", text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) == 1:
            continue
        if _PUNCT_ONLY_LINE_RE.fullmatch(stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def chunk_article(
    text: str,
    encoder: tiktoken.Encoding,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_size: int = MIN_CHUNK_SIZE,
) -> list[tuple[str, int]]:
    tokens = encoder.encode(text)
    if len(tokens) < min_size:
        return []

    stride = chunk_size - overlap
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(tokens):
        piece = tokens[start : start + chunk_size]
        if len(piece) < min_size:
            break
        chunks.append((encoder.decode(piece), len(piece)))
        if start + chunk_size >= len(tokens):
            break
        start += stride
    return chunks


def process_articles(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, float | int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = tiktoken.get_encoding(ENCODING_NAME)

    total_chunks = 0
    total_articles = 0
    token_counts: list[int] = []

    with input_path.open(encoding="utf-8") as in_file, output_path.open("w", encoding="utf-8") as out_file:
        lines = sum(1 for _ in in_file)
        in_file.seek(0)

        for line in tqdm(in_file, total=lines, desc="Chunking articles", unit="article"):
            article = json.loads(line)
            total_articles += 1
            article_id = str(article["id"])
            cleaned = clean_text(article["text"])

            for chunk_index, (chunk_text, token_count) in enumerate(
                chunk_article(cleaned, encoder)
            ):
                record = {
                    "chunk_id": f"{article_id}_{chunk_index}",
                    "article_id": article_id,
                    "title": article["title"],
                    "url": article["url"],
                    "text": chunk_text,
                    "token_count": token_count,
                    "chunk_index": chunk_index,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1
                token_counts.append(token_count)

    if not token_counts:
        return {
            "total_chunks": 0,
            "avg_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "total_articles": total_articles,
        }

    return {
        "total_chunks": total_chunks,
        "avg_tokens": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "total_articles": total_articles,
    }


def print_stats(stats: dict[str, float | int]) -> None:
    print(f"Total articles processed: {stats['total_articles']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Average tokens per chunk: {stats['avg_tokens']:.2f}")
    print(f"Min tokens per chunk: {stats['min_tokens']}")
    print(f"Max tokens per chunk: {stats['max_tokens']}")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()
    logger.info("Reading articles from {}", INPUT_PATH)
    stats = process_articles()
    logger.info("Wrote chunks to {}", OUTPUT_PATH)
    print_stats(stats)


if __name__ == "__main__":
    main()
