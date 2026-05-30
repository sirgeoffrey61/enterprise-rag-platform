"""Production FastAPI application for the Enterprise RAG platform."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.cache import QueryCache
from api.generator import GroundedGenerator, answer_query
from api.schemas import (
    AskRequest,
    AskResponse,
    BenchmarkResponse,
    GpuInfo,
    HealthResponse,
    MetricsResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk,
    ServiceStatus,
    SourceChunk,
)
from evaluation.benchmark import BENCHMARK_PATH, EvaluationRunner
from observability.metrics import MetricsRecorder
from observability.tracer import QueryTracer
from reranking.reranker import CrossEncoderReranker, release_hybrid_gpu
from retrieval.hybrid_retriever import HybridRetriever

load_dotenv(PROJECT_ROOT / ".env")

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class AppState:
    retriever: HybridRetriever | None = None
    reranker: CrossEncoderReranker | None = None
    generator: GroundedGenerator | None = None
    cache: QueryCache | None = None
    metrics: MetricsRecorder | None = None
    tracer: QueryTracer | None = None
    total_queries_served: int = 0


state = AppState()


def _check_qdrant() -> ServiceStatus:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
        client.get_collection("enterprise_rag")
        return ServiceStatus(status="ok")
    except Exception as exc:
        return ServiceStatus(status="error", detail=str(exc))


def _check_redis() -> ServiceStatus:
    try:
        if state.cache is None:
            probe = QueryCache()
            probe.client.ping()
        else:
            state.cache.client.ping()
        return ServiceStatus(status="ok")
    except Exception as exc:
        return ServiceStatus(status="error", detail=str(exc))


def _gpu_info() -> GpuInfo:
    try:
        import torch

        if torch.cuda.is_available():
            return GpuInfo(available=True, device_name=torch.cuda.get_device_name(0))
        return GpuInfo(available=False)
    except Exception:
        return GpuInfo(available=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise RAG API...")
    state.metrics = MetricsRecorder()
    state.tracer = QueryTracer()
    state.cache = QueryCache()
    state.retriever = HybridRetriever()
    state.reranker = CrossEncoderReranker(device="cpu")

    try:
        state.generator = GroundedGenerator()
        logger.info("GroundedGenerator initialized")
    except ValueError as exc:
        state.generator = None
        logger.warning("GroundedGenerator unavailable: {}", exc)

    logger.info("API startup complete")
    yield

    logger.info("Shutting down Enterprise RAG API...")
    if state.retriever is not None:
        release_hybrid_gpu(state.retriever)
        state.retriever = None
    state.reranker = None
    state.generator = None
    state.cache = None
    logger.info("API shutdown complete")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        logger.info("{} {}", request.method, request.url.path)
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "{} {} -> {} ({:.1f}ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


app = FastAPI(
    title="Enterprise RAG Platform",
    description="Production RAG API with hybrid retrieval, reranking, and grounded generation",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    if state.generator is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not configured. /ask requires grounded generation.",
        )
    if state.retriever is None or state.reranker is None or state.cache is None:
        raise HTTPException(status_code=503, detail="RAG services not initialized")

    trace_id = str(uuid.uuid4())
    started = time.perf_counter()

    result = await asyncio.to_thread(
        answer_query,
        body.query,
        cache=state.cache,
        query_id=trace_id,
        candidate_k=body.candidate_k,
        rerank_top_k=body.top_k,
        max_chunks=body.top_k,
        retriever=state.retriever,
        reranker=state.reranker,
        generator=state.generator,
        metrics=state.metrics,
        tracer=state.tracer,
    )

    latency_ms = float(result.get("timings", {}).get("total_ms", 0.0))
    if latency_ms == 0.0:
        latency_ms = (time.perf_counter() - started) * 1000

    state.total_queries_served += 1

    cited_ids = set(result.get("cited_chunk_ids", []))
    sources = [
        SourceChunk(
            chunk_id=chunk["chunk_id"],
            title=chunk["title"],
            url=chunk.get("url"),
            rerank_score=chunk.get("rerank_score"),
            cited=chunk["chunk_id"] in cited_ids,
        )
        for chunk in result.get("raw_chunks", [])
    ]

    return AskResponse(
        answer=result["answer"],
        cited_chunk_ids=result.get("cited_chunk_ids", []),
        grounding_ratio=float(result.get("grounding_ratio", 0.0)),
        cache_hit=bool(result.get("cache_hit", False)),
        latency_ms=latency_ms,
        trace_id=result.get("query_id", trace_id),
        sources=sources,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(body: RetrieveRequest) -> RetrieveResponse:
    if state.retriever is None or state.reranker is None:
        raise HTTPException(status_code=503, detail="Retrieval services not initialized")

    started = time.perf_counter()

    def _run() -> list[dict]:
        candidates = state.retriever.search(  # type: ignore[union-attr]
            body.query, top_k=25, candidate_k=25
        )
        return state.reranker.rerank(body.query, candidates, top_k=body.top_k)  # type: ignore[union-attr]

    ranked = await asyncio.to_thread(_run)
    latency_ms = (time.perf_counter() - started) * 1000
    state.total_queries_served += 1

    chunks = [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            title=row["title"],
            url=row.get("url"),
            text=row["text"],
            rrf_score=row.get("rrf_score"),
            rerank_score=row.get("rerank_score"),
        )
        for row in ranked
    ]

    return RetrieveResponse(query=body.query, chunks=chunks, latency_ms=latency_ms)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    qdrant = await asyncio.to_thread(_check_qdrant)
    redis = await asyncio.to_thread(_check_redis)
    gpu = _gpu_info()

    overall = "ok"
    if qdrant.status != "ok" or redis.status != "ok":
        overall = "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant,
        redis=redis,
        gpu=gpu,
        total_queries_served=state.total_queries_served,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    if state.metrics is None or state.tracer is None:
        raise HTTPException(status_code=503, detail="Metrics not initialized")

    m = state.metrics
    return MetricsResponse(
        total_queries=m.total_queries,
        cache_hit_rate=round(m.cache_hit_rate, 4),
        avg_latency_ms=round(m.avg_latency_ms, 2),
        avg_grounding_ratio=round(m.avg_grounding_ratio, 4),
        insufficient_evidence_rate=round(m.insufficient_evidence_rate, 4),
        last_5_traces=state.tracer.last_traces(5),
    )


def _run_benchmark_with_state() -> dict[str, Any]:
    if state.generator is None:
        raise ValueError("GROQ_API_KEY required to run benchmark")

    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.benchmark_path = BENCHMARK_PATH
    runner.cache = state.cache
    runner.retriever = state.retriever
    runner.reranker = state.reranker
    runner.generator = state.generator
    return runner.run_benchmark(release_gpu_after=False)


@app.get("/benchmark", response_model=BenchmarkResponse)
async def benchmark() -> BenchmarkResponse:
    if state.retriever is None or state.reranker is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    if state.generator is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY required to run benchmark",
        )

    try:
        report = await asyncio.to_thread(_run_benchmark_with_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return BenchmarkResponse(
        generated_at=report["generated_at"],
        total_queries=report["total_queries"],
        metrics=report["metrics"],
        results=report["results"],
    )
