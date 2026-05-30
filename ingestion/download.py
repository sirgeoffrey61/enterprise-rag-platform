"""Download English Wikipedia articles for the Enterprise RAG ingestion pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

ARTICLE_COUNT = 10_000
LOG_EVERY = 1_000
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "wikipedia_raw.jsonl"


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def article_url(title: str) -> str:
    slug = title.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{slug}"


def download_wikipedia_articles(count: int = ARTICLE_COUNT, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset {} ({}) with streaming=True", DATASET_NAME, DATASET_CONFIG)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as out_file:
        progress = tqdm(total=count, desc="Downloading Wikipedia articles", unit="article")
        for row in dataset:
            record = {
                "id": row.get("id", written),
                "title": row["title"],
                "text": row["text"],
                "url": row.get("url") or article_url(row["title"]),
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            progress.update(1)

            if written % LOG_EVERY == 0:
                logger.info("Downloaded {} / {} articles", written, count)

            if written >= count:
                break

        progress.close()

    logger.info("Finished. Saved {} articles to {}", written, output_path)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()
    download_wikipedia_articles()


if __name__ == "__main__":
    main()
