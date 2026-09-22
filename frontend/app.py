import os
import streamlit as st
from api_client import RAGApiClient, APIClientError

# Configure Streamlit Page
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .source-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #BFDBFE;
    }
    .timing-badge {
        font-size: 0.78rem;
        color: #94A3B8;
        text-align: right;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize API Client
@st.cache_resource
def get_client() -> RAGApiClient:
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    return RAGApiClient(base_url=api_url)

client = get_client()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am your **RAG Document Assistant**. Ask me any question grounded in the uploaded Computer Science educational materials (Operating Systems, Database Systems, Computer Networks, and Machine Learning Transformers).",
            "sources": [],
            "citations": [],
            "execution_time_ms": None
        }
    ]

# Sidebar
with st.sidebar:
    st.title("⚙️ System Status")
    
    # Check Backend Health
    health = client.check_health()
    if health.get("connected"):
        data = health.get("data", {})
        status_val = data.get("status", "healthy")
        st.success(f"🟢 **Backend Connected** ({status_val.upper()})")
        
        components = data.get("components", {})
        vs_info = components.get("vector_store", {}).get("details", {})
        ollama_info = components.get("ollama_llm", {}).get("details", {})
        backend_name = ollama_info.get("backend", "LLM")
        
        with st.expander("📊 Vector Store Details", expanded=True):
            st.write(f"• **Collection:** `{vs_info.get('collection_name', 'N/A')}`")
            st.write(f"• **Chunks Stored:** `{vs_info.get('total_chunks', 0)}`")
            st.write(f"• **Embedding Model:** `{vs_info.get('model_name', 'all-MiniLM-L6-v2')}`")

        with st.expander(f"✨ {backend_name} Details", expanded=False):
            st.write(f"• **Backend:** `{ollama_info.get('backend', 'N/A')}`")
            st.write(f"• **Configured Model:** `{ollama_info.get('configured_model', 'N/A')}`")
            st.write(f"• **Host:** `{ollama_info.get('host', 'N/A')}`")
            st.write(f"• **Status:** `{'✅ Connected' if ollama_info.get('model_present') else '⚠️ Not Ready'}`")
    else:
        st.error("🔴 **Backend Offline**")
        st.caption(health.get("error", "Unable to connect to FastAPI backend."))

    st.divider()
    st.subheader("💡 Example Questions")
    example_questions = [
        "What are the four Coffman conditions for a deadlock?",
        "Explain the difference between clustered and secondary indexes.",
        "How does HTTP/3 with QUIC eliminate Head-of-Line blocking?",
        "What is the mathematical formula for Scaled Dot-Product Attention?",
        "What is Dijkstra's Banker's Algorithm used for?",
        "Explain the four ACID transaction properties."
    ]

    for eq in example_questions:
        if st.button(eq, key=f"ex_{hash(eq)}", use_container_width=True):
            st.session_state["prefill_query"] = eq

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I assist you with your documents today?",
                "sources": [],
                "citations": [],
                "execution_time_ms": None
            }
        ]
        st.rerun()


# Main Application Area
st.markdown('<div class="main-title">📚 RAG-Powered Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Grounded Question Answering with Verified Source Citations</div>', unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Sources / Citations if available
        citations = msg.get("citations", [])
        sources = msg.get("sources", [])
        
        if citations:
            st.markdown("##### 📌 Cited Sources")
            citations_html = "".join([f'<span class="source-badge">📄 {c}</span>' for c in citations])
            st.markdown(citations_html, unsafe_allow_html=True)

        if sources:
            with st.expander("🔍 Inspect Retrieved Context Chunks"):
                for idx, src in enumerate(sources, start=1):
                    doc = src.get("document", "Document")
                    page = src.get("page", 1)
                    score = src.get("score")
                    snippet = src.get("snippet", "")
                    score_str = f" | Relevance: {score:.2%}" if score is not None else ""
                    st.markdown(
                        f"""<div class="citation-card">
                        <strong>Source #{idx}: {doc} (Page {page}){score_str}</strong><br>
                        <em>"{snippet}"</em>
                        </div>""",
                        unsafe_allow_html=True
                    )

        if msg.get("execution_time_ms"):
            st.markdown(f'<div class="timing-badge">⚡ Responded in {msg["execution_time_ms"]} ms</div>', unsafe_allow_html=True)

# Determine Query Input
user_input = st.chat_input("Ask a question about your documents...")

# Check if an example question was clicked
if "prefill_query" in st.session_state and st.session_state["prefill_query"]:
    user_input = st.session_state.pop("prefill_query")

# Handle User Submission
if user_input:
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Query Backend
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving context from ChromaDB & generating grounded answer..."):
            try:
                response = client.query(user_input)
                answer = response.get("answer", "No response generated.")
                sources = response.get("sources", [])
                citations = response.get("source_citations", [])
                exec_time = response.get("execution_time_ms")

                st.markdown(answer)

                if citations:
                    st.markdown("##### 📌 Cited Sources")
                    citations_html = "".join([f'<span class="source-badge">📄 {c}</span>' for c in citations])
                    st.markdown(citations_html, unsafe_allow_html=True)

                if sources:
                    with st.expander("🔍 Inspect Retrieved Context Chunks"):
                        for idx, src in enumerate(sources, start=1):
                            doc = src.get("document", "Document")
                            page = src.get("page", 1)
                            score = src.get("score")
                            snippet = src.get("snippet", "")
                            score_str = f" | Relevance: {score:.2%}" if score is not None else ""
                            st.markdown(
                                f"""<div class="citation-card">
                                <strong>Source #{idx}: {doc} (Page {page}){score_str}</strong><br>
                                <em>"{snippet}"</em>
                                </div>""",
                                unsafe_allow_html=True
                            )

                if exec_time:
                    st.markdown(f'<div class="timing-badge">⚡ Responded in {exec_time} ms</div>', unsafe_allow_html=True)

                # Save assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "citations": citations,
                    "execution_time_ms": exec_time
                })

            except APIClientError as err:
                st.error(f"❌ **API Error:** {str(err)}")
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred: {str(e)}")
