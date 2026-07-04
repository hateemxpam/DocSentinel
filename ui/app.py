"""
app.py - Streamlit User Interface for DocSentinel Compliance Assistant
======================================================================
"""

import uuid
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# API Connection settings
API_URL = "http://localhost:8000"

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="DocSentinel — Intelligent Compliance Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Premium CSS Design System ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding: 2.5rem 3rem 3rem 3rem;
    max-width: 1100px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0d1320 100%);
    border-right: 1px solid #1e2a3a;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.4rem;
}

/* ── Sidebar logo area ── */
.ds-logo-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}
.ds-logo-icon {
    font-size: 32px;
    line-height: 1;
}
.ds-logo-text {
    font-size: 20px;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.3px;
}
.ds-logo-sub {
    font-size: 11px;
    color: #5a7a9a;
    margin-top: 2px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Sidebar section labels ── */
.sb-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #3d5a78;
    margin: 20px 0 8px 0;
}

/* ── Status pill ── */
.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: #0f1923;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    margin-bottom: 6px;
}
.status-label { font-size: 13px; color: #7a9bbf; }
.status-ok   { font-size: 12px; font-weight: 600; color: #22c55e; }
.status-fail { font-size: 12px; font-weight: 600; color: #ef4444; }

/* ── Total queries chip ── */
.stat-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: #0f1923;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    margin-bottom: 6px;
}
.stat-chip-icon { font-size: 18px; }
.stat-chip-val  { font-size: 22px; font-weight: 700; color: #58a6ff; line-height: 1; }
.stat-chip-desc { font-size: 11px; color: #5a7a9a; margin-top: 1px; }

/* ── Session chip ── */
.session-chip {
    padding: 8px 12px;
    background: #0f1923;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    font-size: 12px;
    color: #5a7a9a;
}
.session-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #58a6ff;
}

/* ── Main header ── */
.ds-header {
    margin-bottom: 2rem;
}
.ds-title {
    font-size: 36px;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.8px;
    line-height: 1.1;
    margin: 0;
}
.ds-title span { color: #58a6ff; }
.ds-subtitle {
    font-size: 14px;
    color: #5a7a9a;
    margin-top: 6px;
    font-weight: 400;
}
.ds-subtitle b { color: #7a9bbf; font-weight: 500; }

/* ── Search bar wrapper ── */
.search-wrap {
    background: #0f1923;
    border: 1px solid #1e2a3a;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s;
}
.search-wrap:hover { border-color: #2d4a6a; }

/* ── Streamlit input override ── */
.stTextInput > div > div > input {
    background: #141d2b !important;
    color: #e6edf3 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    border: none !important;
    color: #fff !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(31,111,235,0.4) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #1e2a3a !important;
    color: #7a9bbf !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #58a6ff !important;
    color: #58a6ff !important;
}

/* ── Response card ── */
.resp-card {
    background: #0d1117;
    border: 1px solid #1e2a3a;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
}
.resp-status-success {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: #22c55e;
    margin-bottom: 1.2rem;
}
.resp-status-blocked {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: #ef4444;
    margin-bottom: 1.2rem;
}
.resp-status-warning {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: #f59e0b;
    margin-bottom: 1.2rem;
}
.resp-answer {
    font-size: 15px;
    line-height: 1.75;
    color: #c9d1d9;
    white-space: pre-wrap;
}
.conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 16px;
    padding: 7px 16px;
    background: #0f1923;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    font-size: 13px;
    color: #5a7a9a;
}
.conf-value {
    font-weight: 700;
    font-size: 16px;
}

/* ── Divider ── */
.ds-divider {
    border: none;
    border-top: 1px solid #1e2a3a;
    margin: 1.4rem 0;
}

/* ── Recent queries ── */
.recent-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #3d5a78;
    margin-bottom: 10px;
}
.stButton > button.hist-btn {
    text-align: left !important;
    font-weight: 400 !important;
    font-size: 13px !important;
    color: #7a9bbf !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f1923;
    border: 1px dashed #1e2a3a;
    border-radius: 10px;
    padding: 0.5rem;
}

/* ── Streamlit expander ── */
details {
    background: #0f1923 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "last_response" not in st.session_state:
    st.session_state.last_response = None


# --- 4. Query Executor ---
def execute_query(query_text):
    if not query_text.strip():
        st.warning("Please enter a valid question.")
        return

    if query_text not in st.session_state.history:
        st.session_state.history.insert(0, query_text)
        st.session_state.history = st.session_state.history[:5]

    with st.spinner("Analysing documents and generating response…"):
        try:
            payload = {
                "query": query_text,
                "min_confidence": 0.50,
                "session_id": st.session_state.session_id,
            }
            response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
            if response.status_code == 200:
                st.session_state.last_response = response.json()
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                st.session_state.last_response = None
        except Exception as exc:
            st.error(f"Could not reach DocSentinel API: {exc}")
            st.session_state.last_response = None


# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="ds-logo-wrap">
        <span class="ds-logo-icon">🛡️</span>
        <div>
            <div class="ds-logo-text">DocSentinel</div>
            <div class="ds-logo-sub">Compliance Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── System Status ──
    st.markdown('<div class="sb-label">System Status</div>', unsafe_allow_html=True)
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=15)
        if health_resp.status_code == 200:
            h = health_resp.json()
            qdrant_cls  = "status-ok"   if h.get("qdrant")   == "ok" else "status-fail"
            qdrant_txt  = "Online"       if h.get("qdrant")   == "ok" else "Offline"
            postgres_cls = "status-ok"  if h.get("postgres") == "ok" else "status-fail"
            postgres_txt = "Online"      if h.get("postgres") == "ok" else "Offline"
            st.markdown(f"""
            <div class="status-row">
                <span class="status-label">Vector DB (Qdrant)</span>
                <span class="{qdrant_cls}">● {qdrant_txt}</span>
            </div>
            <div class="status-row">
                <span class="status-label">PostgreSQL</span>
                <span class="{postgres_cls}">● {postgres_txt}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-row"><span class="status-label">API</span><span class="status-fail">● Offline</span></div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="status-row"><span class="status-label">API</span><span class="status-fail">● Unreachable</span></div>', unsafe_allow_html=True)

    # ── Total Queries ──
    st.markdown('<div class="sb-label">Usage</div>', unsafe_allow_html=True)
    try:
        stats_resp = requests.get(f"{API_URL}/stats", timeout=10)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            total = stats.get("total_queries", 0)
            st.markdown(f"""
            <div class="stat-chip">
                <span class="stat-chip-icon">💬</span>
                <div>
                    <div class="stat-chip-val">{total:,}</div>
                    <div class="stat-chip-desc">Total Queries Processed</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("Stats unavailable")
    except Exception:
        st.caption("Stats unavailable")

    # ── Upload ──
    st.markdown('<div class="sb-label">Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF only", type=["pdf"], label_visibility="collapsed"
    )
    if uploaded_file is not None:
        if st.button("⚡ Ingest Document", use_container_width=True, type="primary"):
            with st.spinner("Parsing, chunking and embedding…"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    up_resp = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        data={"session_id": st.session_state.session_id},
                        timeout=120
                    )
                    if up_resp.status_code == 200:
                        d = up_resp.json()
                        st.success(f"✅ {d.get('filename')} ingested — {d.get('chunks_stored')} chunks created.")
                    else:
                        st.error(f"Ingestion failed: {up_resp.text}")
                except Exception as exc:
                    st.error(f"Error: {exc}")

    # ── Session ──
    st.markdown('<div class="sb-label">Session</div>', unsafe_allow_html=True)
    short_id = st.session_state.session_id[:8]
    st.markdown(f"""
    <div class="session-chip">
        Session ID &nbsp;<span class="session-id">{short_id}…</span>
    </div>
    <div style="font-size:11px; color:#3d5a78; margin-top:6px; padding: 0 2px;">
        GDPR &amp; EU AI Act available to all users by default.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  MAIN PANEL
# ═══════════════════════════════════════════════

# Header
st.markdown("""
<div class="ds-header">
    <div class="ds-title">🛡️ Doc<span>Sentinel</span></div>
    <div class="ds-subtitle">
        Intelligent Policy Compliance Assistant &nbsp;·&nbsp;
        <b>Llama 3.3 70B</b> &nbsp;·&nbsp; <b>Hybrid RAG</b> &nbsp;·&nbsp; <b>Trust-Gated</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Search bar
with st.container():
    query_input = st.text_input(
        "Query",
        value=st.session_state.query_input,
        placeholder="Ask anything about your compliance documents…",
        label_visibility="collapsed"
    )
    c1, c2, c3 = st.columns([2.2, 1.2, 6.6])
    with c1:
        ask_clicked = st.button("🔍 Ask DocSentinel", type="primary", use_container_width=True)
    with c2:
        if st.button("✕ Clear", type="secondary", use_container_width=True):
            st.session_state.query_input = ""
            st.session_state.last_response = None
            st.rerun()

if ask_clicked:
    st.session_state.query_input = query_input
    execute_query(query_input)

# ── Response ──
if st.session_state.last_response is not None:
    res    = st.session_state.last_response
    status = res.get("status")
    answer = res.get("answer")
    citations = res.get("citations", [])
    avg_conf  = res.get("avg_confidence")
    hallucination_flagged = res.get("hallucination_flagged", False)

    st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
    st.markdown("**Response**", unsafe_allow_html=False)

    # ── Status badge ──
    if status == "SUCCESS":
        badge_cls  = "resp-status-success"
        badge_text = "✅ &nbsp;Answer Found"
    elif status == "HALLUCINATION_RISK":
        badge_cls  = "resp-status-warning"
        badge_text = "⚠️ &nbsp;Hallucination Risk"
    elif status == "BLOCKED":
        badge_cls  = "resp-status-blocked"
        badge_text = "🚫 &nbsp;Blocked — Low Confidence"
    else:
        badge_cls  = "resp-status-warning"
        badge_text = "⚠️ &nbsp;Insufficient Context"

    st.markdown(f'<span class="{badge_cls}">{badge_text}</span>', unsafe_allow_html=True)

    # ── Answer text ──
    if status in ("SUCCESS", "HALLUCINATION_RISK") and answer:
        st.markdown(answer)

        # Confidence badge
        if avg_conf is not None:
            conf_pct = avg_conf * 100
            if conf_pct >= 75:
                conf_color = "#22c55e"
            elif conf_pct >= 55:
                conf_color = "#f59e0b"
            else:
                conf_color = "#ef4444"
            st.markdown(
                f'<div class="conf-badge">Source Confidence &nbsp; '
                f'<span class="conf-value" style="color:{conf_color};">{conf_pct:.1f}%</span></div>',
                unsafe_allow_html=True
            )

        if hallucination_flagged:
            st.warning("⚠️ This response was flagged during the hallucination audit. Review sources carefully.")

        # Citations
        if citations:
            with st.expander("📎 Source Citations"):
                records = []
                for c in citations:
                    ret_at = c.get("retrieved_at", "")
                    try:
                        ret_at = datetime.fromisoformat(ret_at).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        pass
                    records.append({
                        "Source Document": c.get("source"),
                        "Page": c.get("page_number"),
                        "Match Score": f"{c.get('confidence_score', 0.0) * 100:.1f}%",
                        "Retrieved At": ret_at,
                    })
                st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

    elif status == "BLOCKED":
        conf_pct = avg_conf * 100 if avg_conf else 0
        st.info(f"Retrieved sources had an average confidence of **{conf_pct:.1f}%**, below the required 50% threshold. Try rephrasing your question or uploading a more relevant document.")

    elif status == "INSUFFICIENT_CONTEXT":
        st.info("The available documents do not contain enough information to answer this question reliably. Try uploading a relevant document first.")

# ── Recent Queries ──
if st.session_state.history:
    st.markdown('<hr class="ds-divider">', unsafe_allow_html=True)
    st.markdown('<div class="recent-label">Recent Queries</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(st.session_state.history), 3))
    for i, past_query in enumerate(st.session_state.history):
        with cols[i % 3]:
            short = past_query[:55] + "…" if len(past_query) > 55 else past_query
            if st.button(f"↩ {short}", key=f"hist_{i}", use_container_width=True, type="secondary"):
                st.session_state.query_input = past_query
                execute_query(past_query)
                st.rerun()
