"""Redis query cache for Enterprise RAG pipeline results."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

DEFAULT_TTL_SECONDS = 3600
KEY_PREFIX = "enterprise_rag:query:"


class QueryCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        decode_responses: bool = True,
    ) -> None:
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=decode_responses,
            protocol=2,  # compatible with Redis 5.x / alpine via Docker
        )
        self.client.ping()

    @staticmethod
    def _cache_key(query: str) -> str:
        digest = hashlib.md5(query.strip().encode("utf-8")).hexdigest()
        return f"{KEY_PREFIX}{digest}"

    def get(self, query: str) -> dict[str, Any] | None:
        raw = self.client.get(self._cache_key(query))
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, query: str, result: dict[str, Any], ttl: int = DEFAULT_TTL_SECONDS) -> None:
        payload = json.dumps(result, ensure_ascii=False)
        self.client.setex(self._cache_key(query), ttl, payload)

    def get_stats(self) -> dict[str, int]:
        total_keys = 0
        for _key in self.client.scan_iter(match=f"{KEY_PREFIX}*"):
            total_keys += 1
        return {"total_keys": total_keys}
