"""Streamlit frontend for the Enterprise RAG platform."""

from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Any

import httpx
import streamlit as st

API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
INSUFFICIENT_MARKER = "INSUFFICIENT EVIDENCE"
CITATION_RE = re.compile(r"\[(\d+)\]")
TIMEOUT = httpx.Timeout(300.0, connect=10.0)

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .main-header {
        font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #1e3a5f 0%, #2563eb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .answer-box {
        background: #f8fafc; border-left: 4px solid #2563eb;
        padding: 1.25rem 1.5rem; border-radius: 8px;
        line-height: 1.65; font-size: 1.02rem;
    }
    .cite {
        background: #dbeafe; color: #1d4ed8;
        padding: 0.1rem 0.35rem; border-radius: 4px;
        font-weight: 600; font-size: 0.9em;
    }
    .source-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem 1.1rem;
        margin-bottom: 0.75rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .source-card.cited { border-color: #2563eb; background: #eff6ff; }
    .badge-high { background: #dcfce7; color: #166534; padding: 0.25rem 0.75rem;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem; }
    .badge-medium { background: #fef9c3; color: #854d0e; padding: 0.25rem 0.75rem;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem; }
    .badge-low { background: #fee2e2; color: #991b1b; padding: 0.25rem 0.75rem;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem; }
    .grounding-bar {
        height: 12px; border-radius: 6px; margin: 0.5rem 0 0.25rem 0;
    }
    .status-ok { color: #16a34a; font-weight: 600; }
    .status-error { color: #dc2626; font-weight: 600; }
    .preview-text { color: #475569; font-size: 0.9rem; line-height: 1.5; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def api_get(path: str) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(f"{API_BASE}{path}")
        response.raise_for_status()
        return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(f"{API_BASE}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def grounding_color(ratio: float) -> str:
    if ratio >= 0.6:
        return "#22c55e"
    if ratio >= 0.3:
        return "#eab308"
    return "#ef4444"


def confidence_level(answer: str, ratio: float) -> tuple[str, str]:
    if INSUFFICIENT_MARKER in answer:
        return "Low", "badge-low"
    if ratio >= 0.6:
        return "High", "badge-high"
    if ratio >= 0.3:
        return "Medium", "badge-medium"
    return "Low", "badge-low"


def highlight_citations(answer: str) -> str:
    escaped = (
        answer.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    return CITATION_RE.sub(r'<span class="cite">[\1]</span>', escaped)


def render_grounding_bar(ratio: float) -> None:
    pct = max(0.0, min(1.0, ratio))
    color = grounding_color(pct)
    st.markdown(
        f"""
        <div class="grounding-bar" style="width:100%; background:#e2e8f0;">
            <div style="width:{pct * 100:.1f}%; height:12px; background:{color};
                border-radius:6px;"></div>
        </div>
        <span style="color:#64748b; font-size:0.85rem;">
            Grounding ratio: {pct:.0%}</span>
        """,
        unsafe_allow_html=True,
    )


def render_source_card(
    *,
    title: str,
    url: str | None,
    rerank_score: float | None,
    cited: bool = False,
    text_preview: str | None = None,
    rrf_score: float | None = None,
) -> None:
    cited_class = " cited" if cited else ""
    url_html = (
        f'<a href="{url}" target="_blank" rel="noopener">{url}</a>'
        if url
        else "<span style='color:#94a3b8'>No URL</span>"
    )
    scores = []
    if rerank_score is not None:
        scores.append(f"Rerank: {rerank_score:.3f}")
    if rrf_score is not None:
        scores.append(f"RRF: {rrf_score:.4f}")
    score_line = " · ".join(scores) if scores else ""
    preview = ""
    if text_preview:
        preview = f'<p class="preview-text">{text_preview[:400]}...</p>' if len(text_preview) > 400 else f'<p class="preview-text">{text_preview}</p>'
    cite_badge = '<span style="color:#2563eb;font-weight:600;"> ✓ Cited</span>' if cited else ""
    st.markdown(
        f"""
        <div class="source-card{cited_class}">
            <strong>{title}</strong>{cite_badge}<br>
            {url_html}<br>
            <span style="color:#64748b;font-size:0.8rem;">{score_line}</span>
            {preview}
        </div>
        """,
        unsafe_allow_html=True,
    )


def tab_ask() -> None:
    st.subheader("Ask a question")
    st.caption("Grounded answers with citations from your enterprise knowledge base.")

    query = st.text_input("Your question", placeholder="e.g. What is anarchism?", key="ask_query")
    top_k = st.slider("Number of sources (top_k)", min_value=1, max_value=10, value=5, key="ask_top_k")
    submit = st.button("Get answer", type="primary", use_container_width=False)

    if not submit:
        return
    if not query.strip():
        st.warning("Please enter a question.")
        return

    with st.spinner("Searching, reranking, and generating answer…"):
        try:
            data = api_post("/ask", {"query": query.strip(), "top_k": top_k})
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            st.error(f"API error ({exc.response.status_code}): {detail}")
            return
        except httpx.ConnectError:
            st.error(f"Cannot reach API at {API_BASE}. Start the server with uvicorn.")
            return
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return

    answer = data["answer"]
    ratio = float(data["grounding_ratio"])
    label, badge_class = confidence_level(answer, ratio)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<span class="{badge_class}">Confidence: {label}</span>', unsafe_allow_html=True)
    with col2:
        cache_label = "✓ Cache hit" if data["cache_hit"] else "Cache miss"
        st.metric("Cache", cache_label)
    with col3:
        st.metric("Latency", f"{data['latency_ms']:.0f} ms")
    with col4:
        st.metric("Trace ID", data["trace_id"][:8] + "…")

    st.markdown("#### Grounding")
    render_grounding_bar(ratio)

    st.markdown("#### Answer")
    st.markdown(f'<div class="answer-box">{highlight_citations(answer)}</div>', unsafe_allow_html=True)

    sources = data.get("sources", [])
    if sources:
        st.markdown("#### Sources")
        for src in sources:
            render_source_card(
                title=src["title"],
                url=src.get("url"),
                rerank_score=src.get("rerank_score"),
                cited=src.get("cited", False),
            )


def tab_retrieve() -> None:
    st.subheader("Retrieve only")
    st.caption("Hybrid search + reranking without LLM generation.")

    query = st.text_input("Search query", placeholder="e.g. Roman Empire fall", key="retrieve_query")
    submit = st.button("Retrieve chunks", type="primary")

    if not submit:
        return
    if not query.strip():
        st.warning("Please enter a search query.")
        return

    with st.spinner("Retrieving top chunks…"):
        try:
            data = api_post("/retrieve", {"query": query.strip(), "top_k": 5})
        except httpx.HTTPStatusError as exc:
            st.error(f"API error ({exc.response.status_code}): {exc.response.text}")
            return
        except httpx.ConnectError:
            st.error(f"Cannot reach API at {API_BASE}.")
            return
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return

    st.metric("Latency", f"{data['latency_ms']:.0f} ms")
    st.markdown(f"**{len(data['chunks'])}** results for: _{data['query']}_")

    for chunk in data["chunks"]:
        render_source_card(
            title=chunk["title"],
            url=chunk.get("url"),
            rerank_score=chunk.get("rerank_score"),
            rrf_score=chunk.get("rrf_score"),
            text_preview=chunk.get("text", ""),
        )


@st.fragment(run_every=timedelta(seconds=30))
def health_panel() -> None:
    try:
        data = api_get("/health")
    except Exception as exc:
        st.error(f"Health check failed: {exc}")
        return

    status = data.get("status", "unknown")
    indicator = "🟢" if status == "ok" else "🟠"
    st.markdown(f"**Overall:** {indicator} {status.upper()}")

    c1, c2, c3 = st.columns(3)
    with c1:
        q = data["qdrant"]
        cls = "status-ok" if q["status"] == "ok" else "status-error"
        st.markdown(f"**Qdrant**<br><span class='{cls}'>{q['status'].upper()}</span>", unsafe_allow_html=True)
        if q.get("detail"):
            st.caption(q["detail"])
    with c2:
        r = data["redis"]
        cls = "status-ok" if r["status"] == "ok" else "status-error"
        st.markdown(f"**Redis**<br><span class='{cls}'>{r['status'].upper()}</span>", unsafe_allow_html=True)
        if r.get("detail"):
            st.caption(r["detail"])
    with c3:
        gpu = data["gpu"]
        if gpu.get("available"):
            st.markdown(f"**GPU**<br><span class='status-ok'>Available</span>", unsafe_allow_html=True)
            st.caption(gpu.get("device_name", ""))
        else:
            st.markdown("**GPU**<br><span class='status-error'>Unavailable</span>", unsafe_allow_html=True)

    st.metric("Total queries served", data.get("total_queries_served", 0))
    st.caption("Auto-refreshes every 30 seconds.")


def tab_health() -> None:
    st.subheader("System health")
    health_panel()


def tab_metrics() -> None:
    st.subheader("Metrics dashboard")
    if st.button("Refresh metrics"):
        st.session_state.pop("metrics_data", None)

    try:
        data = api_get("/metrics")
    except httpx.ConnectError:
        st.error(f"Cannot reach API at {API_BASE}.")
        return
    except Exception as exc:
        st.error(f"Failed to load metrics: {exc}")
        return

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total queries", data["total_queries"])
    with m2:
        st.metric("Cache hit rate", f"{data['cache_hit_rate'] * 100:.1f}%")
    with m3:
        st.metric("Avg latency", f"{data['avg_latency_ms']:.0f} ms")
    with m4:
        st.metric("Avg grounding", f"{data['avg_grounding_ratio']:.0%}")

    st.metric(
        "Insufficient evidence rate",
        f"{data['insufficient_evidence_rate'] * 100:.1f}%",
    )

    traces = data.get("last_5_traces", [])
    st.markdown("#### Recent traces")
    if not traces:
        st.info("No traces recorded yet. Run a query from the Ask tab.")
        return

    for trace in reversed(traces):
        qid = trace.get("query_id", "unknown")
        query_text = trace.get("query", "")
        with st.expander(f"{qid[:8]}… — {query_text[:60]}{'…' if len(query_text) > 60 else ''}"):
            st.json(trace)


def main() -> None:
    st.markdown('<p class="main-header">Enterprise RAG Platform</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Hybrid retrieval · cross-encoder reranking · grounded generation</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### API")
        st.text_input("Base URL", value=API_BASE, disabled=True, help="Set RAG_API_URL env to override")
        st.caption(f"Connected to `{API_BASE}`")

    tab1, tab2, tab3, tab4 = st.tabs(["Ask", "Retrieve Only", "System Health", "Metrics Dashboard"])

    with tab1:
        tab_ask()
    with tab2:
        tab_retrieve()
    with tab3:
        tab_health()
    with tab4:
        tab_metrics()


if __name__ == "__main__":
    main()
