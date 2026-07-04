"""
app.py - Streamlit User Interface for DocSentinel Compliance Assistant
======================================================================
"""

import uuid
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# API Connection settings
API_URL = "http://localhost:8000"

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="DocSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern, premium HSL tailored dark-mode styles
st.markdown("""
<style>
    /* Premium layout styles */
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
    }
    .stTextInput>div>div>input {
        background-color: #1a1f2c;
        color: #ffffff;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 15px;
    }
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 12px 18px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Generate a unique session_id for this browser session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# --- Helper to Trigger Query Run ---
def execute_query(query_text):
    if not query_text.strip():
        st.warning("Please enter a valid question.")
        return

    # Add to history (keep top 5 unique)
    if query_text not in st.session_state.history:
        st.session_state.history.insert(0, query_text)
        st.session_state.history = st.session_state.history[:5]

    with st.spinner("🔍 Searching documents and verifying trust..."):
        try:
            payload = {
                "query": query_text,
                "min_confidence": 0.50,
                "session_id": st.session_state.session_id,
            }
            response = requests.post(f"{API_URL}/query", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.last_response = data
            else:
                st.error(f"API Error (HTTP {response.status_code}): {response.text}")
                st.session_state.last_response = None
        except Exception as exc:
            st.error(f"Failed to connect to DocSentinel API at {API_URL}: {exc}")
            st.session_state.last_response = None


# --- 3. Sidebar Panel ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.title("DocSentinel Panel")
    st.markdown("---")

    # A. System Status Section
    st.subheader("🌐 System Status")
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=15)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            qdrant_status = "✅ Connected" if health_data.get("qdrant") == "ok" else "❌ Failed"
            postgres_status = "✅ Connected" if health_data.get("postgres") == "ok" else "❌ Failed"
            
            st.markdown(f"**Qdrant:** {qdrant_status}")
            st.markdown(f"**PostgreSQL:** {postgres_status}")
        else:
            st.error("❌ API Health Check Failed")
    except Exception:
        st.error("❌ DocSentinel API is offline")

    st.markdown("---")

    # B. Statistics Section
    st.subheader("📈 Usage Statistics")
    try:
        stats_resp = requests.get(f"{API_URL}/stats", timeout=15)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            st.markdown(f"**Total Queries:** {stats.get('total_queries', 0)}")
            st.markdown(f"**Cache Hit Rate:** {stats.get('cache_hit_rate', 0.0) * 100:.1f}%")
            st.markdown(f"**Avg Latency:** {stats.get('avg_latency_ms', 0.0):.0f} ms")
        else:
            st.caption("Unable to load stats")
    except Exception:
        st.caption("API Offline — statistics unavailable")

    st.markdown("---")

    # C. File Uploader Section
    st.subheader("📥 Upload Policy Documents")
    uploaded_file = st.file_uploader("Select a policy document (PDF only)", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Process & Ingest Document", use_container_width=True):
            with st.spinner("⏳ Parsing, chunking and embedding document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    up_resp = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        data={"session_id": st.session_state.session_id},
                        timeout=120
                    )
                    
                    if up_resp.status_code == 200:
                        up_data = up_resp.json()
                        st.success(f"Ingested '{up_data.get('filename')}' successfully! Created {up_data.get('chunks_stored')} chunks.")
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {up_resp.text}")
                except Exception as exc:
                    st.error(f"Error connecting to ingestion service: {exc}")

    st.markdown("---")

    # D. Session Management
    st.subheader("🗂️ My Session")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    st.caption("GDPR & EU AI Act are always available to all users.")
    if st.button("🗑️ Clear My Documents", use_container_width=True, type="secondary"):
        try:
            del_resp = requests.delete(
                f"{API_URL}/session/{st.session_state.session_id}", timeout=10
            )
            if del_resp.status_code == 200:
                data = del_resp.json()
                st.success(f"Cleared {data.get('chunks_deleted', 0)} chunks from your session.")
                # Generate a fresh session_id
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
            else:
                st.error("Failed to clear session.")
        except Exception as exc:
            st.error(f"Error: {exc}")

    st.markdown("---")

    # E. Cache Management
    st.subheader("🧹 Cache Management")
    if st.button("Clear Semantic Cache", use_container_width=True, type="secondary"):
        try:
            del_resp = requests.delete(f"{API_URL}/cache", timeout=5)
            if del_resp.status_code == 200:
                st.success("Semantic cache cleared successfully!")
                st.rerun()
            else:
                st.error("Failed to clear cache.")
        except Exception as exc:
            st.error(f"Error connecting to cache service: {exc}")


# --- 4. Main Panel Area ---
# Header
st.title("🛡️ DocSentinel")
st.markdown("##### Intelligent Policy Compliance Assistant | *Powered by Meta Llama 3.3 70B & Hybrid RAG*")
st.markdown("---")

# Query input
query_input = st.text_input(
    "Ask a compliance question...", 
    value=st.session_state.query_input,
    placeholder="e.g., What are the administrative fines for GDPR violations?"
)

col1, col2 = st.columns([1.5, 8.5])
with col1:
    ask_clicked = st.button("🚀 Ask DocSentinel", use_container_width=True, type="primary")
with col2:
    if st.button("🧹 Clear Input", use_container_width=False):
        st.session_state.query_input = ""
        st.rerun()

if ask_clicked:
    st.session_state.query_input = query_input
    execute_query(query_input)

# --- 5. Display Pipeline Response ---
if "last_response" in st.session_state and st.session_state.last_response is not None:
    res = st.session_state.last_response
    status = res.get("status")
    answer = res.get("answer")
    citations = res.get("citations", [])
    avg_conf = res.get("avg_confidence")
    latency = res.get("latency_ms", 0.0)
    cached = res.get("cached", False)
    hallucination_flagged = res.get("hallucination_flagged", False)

    st.markdown("### Response Analysis")

    # A. SUCCESS
    if status == "SUCCESS":
        st.success("#### ✅ Answer Found")
        st.write(answer)
        
        if hallucination_flagged:
            st.warning("⚠️ **Warning:** This response was flagged for review. A potential hallucination discrepancy was detected between the answer and raw sources.")

    # B. HALLUCINATION_RISK
    elif status == "HALLUCINATION_RISK":
        st.error("#### ⚠️ Hallucination Risk Detected")
        st.error("**CRITICAL SAFETY WARNING:** This response failed consistency audits. Below is the generated response, but it may contain hallucinated or unverified details.")
        st.write(answer)

    # C. BLOCKED
    elif status == "BLOCKED":
        st.error("#### 🚫 Query Blocked")
        st.error("DocSentinel aborted generation because the retrieved source documents did not meet the required trustworthiness threshold.")
        st.info(f"**Average Source Chunk Confidence:** {avg_conf * 100:.1f}% (Minimum threshold is 50.0%)")

    # D. INSUFFICIENT_CONTEXT
    elif status == "INSUFFICIENT_CONTEXT":
        st.warning("#### ⚠️ Insufficient Information")
        st.info("The available compliance policy documents do not contain enough information to formulate a reliable response to this query.")

    # --- Metadata & Citation displays (for SUCCESS and HALLUCINATION_RISK) ---
    if status in ["SUCCESS", "HALLUCINATION_RISK"] and answer:
        # Confidence score only — clean single metric
        if avg_conf is not None:
            conf_pct = avg_conf * 100
            if conf_pct >= 75:
                conf_color = "#22c55e"  # green
            elif conf_pct >= 55:
                conf_color = "#f59e0b"  # amber
            else:
                conf_color = "#ef4444"  # red

            st.markdown(
                f"""<div style="margin-top:12px; display:inline-block; background:#161b22;
                border:1px solid #30363d; border-radius:8px; padding:8px 18px;">
                <span style="color:#8b949e; font-size:13px;">Source Confidence&nbsp;&nbsp;</span>
                <span style="color:{conf_color}; font-size:18px; font-weight:700;">{conf_pct:.1f}%</span>
                </div>""",
                unsafe_allow_html=True
            )


        # Citations Expander
        if citations:
            with st.expander("📎 Sources & Reference Citations"):
                citation_records = []
                for c in citations:
                    # Clean timestamp format
                    ret_at = c.get("retrieved_at", "")
                    if ret_at:
                        try:
                            ret_at = datetime.fromisoformat(ret_at).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass
                    
                    citation_records.append({
                        "Citation Index": c.get("reference"),
                        "Source Document": c.get("source"),
                        "Page Number": c.get("page_number"),
                        "Match Score": f"{c.get('confidence_score', 0.0) * 100:.1f}%",
                        "Retrieved At": ret_at
                    })
                
                df = pd.DataFrame(citation_records)
                st.dataframe(df, use_container_width=True, hide_index=True)


# --- 6. Recent Queries Section ---
if st.session_state.history:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("🕒 Recent Queries")
    for past_query in st.session_state.history:
        if st.button(f"🔍 {past_query}", key=f"hist_{past_query}"):
            st.session_state.query_input = past_query
            execute_query(past_query)
            st.rerun()
