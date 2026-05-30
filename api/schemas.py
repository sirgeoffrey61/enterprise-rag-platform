"""Pydantic request/response models for the Enterprise RAG API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    candidate_k: int = Field(default=25, ge=1, le=100)


class SourceChunk(BaseModel):
    chunk_id: str
    title: str
    url: str | None = None
    rerank_score: float | None = None
    cited: bool = False


class AskResponse(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    grounding_ratio: float
    cache_hit: bool
    latency_ms: float
    trace_id: str
    sources: list[SourceChunk] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)


class RetrievedChunk(BaseModel):
    chunk_id: str
    title: str
    url: str | None = None
    text: str
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float


class ServiceStatus(BaseModel):
    status: str
    detail: str | None = None


class GpuInfo(BaseModel):
    available: bool
    device_name: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: ServiceStatus
    redis: ServiceStatus
    gpu: GpuInfo
    total_queries_served: int


class MetricsResponse(BaseModel):
    total_queries: int
    cache_hit_rate: float
    avg_latency_ms: float
    avg_grounding_ratio: float
    insufficient_evidence_rate: float
    last_5_traces: list[dict[str, Any]]


class BenchmarkResponse(BaseModel):
    generated_at: float
    total_queries: int
    metrics: dict[str, Any]
    results: list[dict[str, Any]]
