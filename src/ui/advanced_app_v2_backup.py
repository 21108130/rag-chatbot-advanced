

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config.settings import settings
from src.auth.auth_manager import get_auth_manager
from src.retrieval.advanced_rag_pipeline import AdvancedRAGPipeline
from src.utils.logger import logger
from src.utils.models import ChatRequest, DocumentStatus


st.set_page_config(
    page_title = "RAG Chatbot v2 — Advanced",
    page_icon  = "🚀",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; margin: 0 2px;
}
.badge-hybrid  { background: #ebf4ff; color: #3182ce; }
.badge-reranked { background: #f0fff4; color: #38a169; }
.badge-stream  { background: #fff5f5; color: #e53e3e; }
.chat-user {
    background: #EBF8FF; border-left: 4px solid #3182CE;
    padding: 0.75rem 1rem; border-radius: 0.5rem; margin: 0.5rem 0;
}
.chat-assistant {
    background: #F0FFF4; border-left: 4px solid #38A169;
    padding: 0.75rem 1rem; border-radius: 0.5rem; margin: 0.5rem 0;
}
.source-box {
    background: #FFFFF0; border: 1px solid #ECC94B;
    border-radius: 0.35rem; padding: 0.5rem 0.75rem;
    font-size: 0.8rem; margin-top: 0.25rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border: 1px solid #667eea44; border-radius: 0.75rem;
    padding: 1rem; text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 800; color: #2D3748; }
.metric-label { font-size: 0.8rem; color: #718096; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)




def init_state() -> None:
    defaults = {
        "authenticated":  False,
        "user_id":        None,
        "username":       None,
        "pipeline":       None,
        "messages":       [],
        "conversation_id": None,
        "indexed_docs":   [],
        "agent_mode":     False,
        "use_streaming":  True,
        "show_metrics":   False,
        "last_response":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v



_PIPELINE_CACHE: dict = {}

def get_pipeline() -> AdvancedRAGPipeline:
    """
    Return the pipeline for the current user.
    Stored at MODULE level (not session state) so it survives all reruns,
    toggle changes, and page interactions without losing indexed documents.
    """
    user_id = st.session_state.get("user_id", "anon")
    if user_id not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[user_id] = AdvancedRAGPipeline(
            enable_reranking   = True,
            enable_streaming   = True,
            max_context_tokens = 3000,
        )

    st.session_state.pipeline = _PIPELINE_CACHE[user_id]
    return _PIPELINE_CACHE[user_id]




def render_auth_page() -> None:
    st.markdown('<div class="main-header">🚀 RAG Chatbot v2</div>', unsafe_allow_html=True)
    st.markdown("#### Advanced AI • Multi-User • Hybrid Search • Streaming")
    st.divider()

    tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

    auth = get_auth_manager()

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary", use_container_width=True):
            if username and password:
                user = auth.authenticate_user(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id       = user.id
                    st.session_state.username      = user.username
                    st.success(f"Welcome back, {user.username}! 🎉")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            else:
                st.warning("Please enter username and password.")

    with tab_register:
        reg_user  = st.text_input("Username", key="reg_user")
        reg_email = st.text_input("Email",    key="reg_email")
        reg_pass  = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create Account", type="primary", use_container_width=True):
            if reg_user and reg_email and reg_pass:
                try:
                    user = auth.create_user(reg_user, reg_email, reg_pass)
                    st.success(f"Account created! Please login as '{reg_user}'.")
                except ValueError as e:
                    st.error(str(e))
            else:
                st.warning("Please fill in all fields.")




def render_sidebar() -> None:
    with st.sidebar:

        st.markdown(f"👤 **{st.session_state.username}**")
        if st.button("Logout", use_container_width=True):

            user_id = st.session_state.get("user_id")
            if user_id and user_id in _PIPELINE_CACHE:
                del _PIPELINE_CACHE[user_id]

            for key in ["authenticated", "user_id", "username", "pipeline",
                        "messages", "conversation_id", "indexed_docs", "last_response"]:
                st.session_state[key] = None if key not in ("messages", "indexed_docs") else []
            st.session_state.authenticated = False
            st.rerun()

        st.divider()


        st.subheader("⚙️ Settings")
        st.session_state.use_streaming = st.toggle(
            "⚡ Streaming responses",
            value = st.session_state.use_streaming,
            help  = "Stream tokens in real-time (ChatGPT-style)",
        )
        st.session_state.agent_mode = st.toggle(
            "🤖 Agent Mode",
            value = st.session_state.agent_mode,
            help  = "Enable multi-step reasoning with web search",
        )
        st.session_state.show_metrics = st.toggle(
            "📊 Show Metrics",
            value = st.session_state.show_metrics,
        )


        try:
            from src.llm.multi_provider_client import get_llm_client
            provider = get_llm_client().get_active_provider()
            icons = {"groq": "⚡", "gemini": "🔵", "openrouter": "🔀"}
            st.caption(f"{icons.get(provider,'🤖')} LLM: **{provider}**")
        except Exception:
            pass
        st.divider()


        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type                  = settings.allowed_extensions,
            accept_multiple_files = True,
        )
        if uploaded_files:
            if st.button("🚀 Index Documents", type="primary", use_container_width=True):
                handle_upload(uploaded_files)


        if st.session_state.indexed_docs:
            st.divider()
            st.subheader("📚 Knowledge Base")
            for doc in st.session_state.indexed_docs:
                icon = "✅" if doc.status == DocumentStatus.INDEXED else "❌"
                st.markdown(
                    f"{icon} **{doc.filename}** <small>({doc.chunk_count} chunks)</small>",
                    unsafe_allow_html=True,
                )


        st.divider()
        render_kb_stats()

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages        = []
            st.session_state.conversation_id = None
            st.session_state.last_response   = None
            st.rerun()


def handle_upload(uploaded_files) -> None:
    pipeline    = get_pipeline()
    upload_path = settings.upload_path

    with st.spinner("Indexing documents …"):
        for uf in uploaded_files:
            if uf.size > settings.max_file_size_bytes:
                st.sidebar.error(f"❌ '{uf.name}' exceeds {settings.max_file_size_mb}MB.")
                continue

            dest = upload_path / uf.name
            dest.write_bytes(uf.getbuffer())

            result = pipeline.ingest_document(dest, user_id=st.session_state.user_id)
            st.session_state.indexed_docs.append(result)

            if result.status == DocumentStatus.INDEXED:
                st.sidebar.success(f"✅ **{uf.name}** ({result.chunk_count} chunks)")
            else:
                st.sidebar.error(f"❌ {uf.name}: {result.message}")


def render_kb_stats() -> None:
    pipeline = get_pipeline()
    try:
        stats = pipeline.get_kb_stats(user_id=st.session_state.user_id)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{stats["total_docs"]}</div>'
                f'<div class="metric-label">Docs</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{stats["total_chunks"]}</div>'
                f'<div class="metric-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("Stats unavailable")



def render_metrics_dashboard() -> None:
    pipeline = get_pipeline()
    with st.expander("📊 Performance Metrics", expanded=True):
        metrics  = pipeline.get_metrics_summary()
        feedback = pipeline.get_feedback_summary(user_id=st.session_state.user_id)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries",    metrics["total_queries"])
        with col2:
            st.metric("Avg Latency",      f"{metrics['avg_latency_ms']:.0f}ms")
        with col3:
            st.metric("Avg Tokens",       f"{metrics['avg_tokens']:.0f}")
        with col4:
            st.metric("Satisfaction",     f"{feedback.get('satisfaction_rate', 0):.0f}%")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("P95 Latency",      f"{metrics['p95_latency_ms']:.0f}ms")
        with col6:
            st.metric("Error Rate",       f"{metrics['error_rate']:.1f}%")
        with col7:
            st.metric("Total Tokens",     metrics["total_tokens"])
        with col8:
            prom = "✅" if metrics["prometheus_enabled"] else "❌"
            st.metric("Prometheus",       prom)


        bad = feedback.get("bad_queries", [])
        if bad:
            st.markdown("**🔴 Recent Low-Rated Queries:**")
            for bq in bad[:5]:
                st.markdown(
                    f"- `{bq['query']}`"
                    + (f" — *{bq['comment']}*" if bq.get("comment") else "")
                )



def render_main() -> None:
    st.markdown('<div class="main-header">🚀 RAG Chatbot v2 — Advanced</div>', unsafe_allow_html=True)


    badges = []
    if st.session_state.use_streaming:
        badges.append('<span class="badge badge-stream">⚡ Streaming</span>')
    if st.session_state.agent_mode:
        badges.append('<span class="badge badge-reranked">🤖 Agent Mode</span>')
    badges.append('<span class="badge badge-hybrid">🔀 Hybrid Search</span>')
    badges.append('<span class="badge badge-reranked">🎯 Reranking</span>')
    st.markdown("&nbsp;".join(badges), unsafe_allow_html=True)

    st.divider()


    if st.session_state.show_metrics:
        render_metrics_dashboard()


    if st.session_state.last_response:
        lr = st.session_state.last_response
        st.markdown("**Rate the last response:**")
        col_up, col_down, col_comment = st.columns([1, 1, 6])
        with col_up:
            if st.button("👍", help="Good response"):
                get_pipeline().record_feedback(
                    user_id         = st.session_state.user_id,
                    query           = lr["query"],
                    answer          = lr["answer"],
                    rating          = 1,
                    conversation_id = st.session_state.conversation_id,
                    latency_ms      = lr.get("latency_ms"),
                )
                st.session_state.last_response = None
                st.success("Thanks for your feedback! 🎉")
                st.rerun()
        with col_down:
            if st.button("👎", help="Bad response"):
                get_pipeline().record_feedback(
                    user_id         = st.session_state.user_id,
                    query           = lr["query"],
                    answer          = lr["answer"],
                    rating          = -1,
                    conversation_id = st.session_state.conversation_id,
                    latency_ms      = lr.get("latency_ms"),
                )
                st.session_state.last_response = None
                st.info("Thanks! We'll work to improve this. 📈")
                st.rerun()

        st.divider()


    if not st.session_state.messages:
        st.info(
            "👈 Upload a document in the sidebar, then ask questions below. "
            "Enable **Agent Mode** for web search + multi-step reasoning.",
            icon="💡",
        )

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">🧑 <strong>You:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-assistant">🤖 <strong>Assistant:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s)", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        fname   = src.metadata.get("source", src.doc_id) if hasattr(src, "metadata") else "Unknown"
                        score   = src.similarity_score if hasattr(src, "similarity_score") else 0.0
                        reranked = src.metadata.get("reranked", False) if hasattr(src, "metadata") else False
                        method   = src.metadata.get("retrieval_method", "vector") if hasattr(src, "metadata") else "vector"
                        preview  = (src.content[:300] if hasattr(src, "content") else str(src)[:300]).replace("\n", " ")
                        st.markdown(
                            f'<div class="source-box">'
                            f"<strong>Source {i}:</strong> {fname} "
                            f"(score: {score:.3f}"
                            + (f" | ✓ reranked" if reranked else "")
                            + f" | {method})<br><em>{preview}…</em></div>",
                            unsafe_allow_html=True,
                        )


            if msg.get("agent_steps"):
                with st.expander(f"🔍 Agent reasoning ({len(msg['agent_steps'])} steps)", expanded=False):
                    for step in msg["agent_steps"]:
                        st.markdown(
                            f"**Step {step['iteration']}:** Called `{step['tool']}` "
                            f"with `{step['args']}`\n\n"
                            f"> {step['observation']}"
                        )

            if msg.get("latency_ms"):
                st.caption(f"⏱ {msg['latency_ms']:.0f}ms")


    with st.form("chat_form", clear_on_submit=True):
        placeholder = "Ask a question…" if not st.session_state.agent_mode else \
                     "Ask anything — I can search your docs AND the web!"
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input("Question", placeholder=placeholder, label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("Send 💬", type="primary")

    if submitted and user_input.strip():
        handle_chat(user_input.strip())
        st.rerun()


def handle_chat(query: str) -> None:
    pipeline = get_pipeline()
    user_id  = st.session_state.user_id


    if st.session_state.agent_mode:
        try:
            from src.agents.rag_agent import RAGAgent

            stats = pipeline.get_kb_stats(user_id=user_id)
            has_docs = stats.get("total_chunks", 0) > 0
            agent = RAGAgent(pipeline=pipeline, user_id=user_id, has_docs=has_docs)
            with st.spinner("🤖 Agent reasoning…"):
                result = agent.run(query)
            answer      = result["answer"]
            agent_steps = result["steps"]
            tools_used  = result["tools_used"]
        except Exception as exc:
            answer      = f"Agent error: {exc}"
            agent_steps = []
            tools_used  = []

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({
            "role":        "assistant",
            "content":     answer,
            "agent_steps": agent_steps,
        })
        st.session_state.last_response = {"query": query, "answer": answer}
        return


    try:
        stats = pipeline.get_kb_stats(user_id=user_id)
        total_chunks = stats.get("total_chunks", 0)
    except Exception:
        total_chunks = 0

    if total_chunks == 0:
        st.warning(
            "⚠️ No documents found in your knowledge base. "
            "Please upload a document using the sidebar, then ask your question again."
        )
        return

    st.session_state.messages.append({"role": "user", "content": query})

    if st.session_state.use_streaming:

        with st.chat_message("assistant"):
            placeholder  = st.empty()
            full_answer  = ""
            start        = __import__("time").perf_counter()

            for token in pipeline.chat_stream(
                query           = query,
                user_id         = user_id,
                conversation_id = st.session_state.conversation_id,
            ):
                full_answer += token
                placeholder.markdown(full_answer + "▌")

            placeholder.markdown(full_answer)
            latency_ms = (__import__("time").perf_counter() - start) * 1000
            st.caption(f"⏱ {latency_ms:.0f}ms | ⚡ streamed")

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    full_answer,
            "latency_ms": latency_ms,
        })
        st.session_state.last_response = {"query": query, "answer": full_answer, "latency_ms": latency_ms}
        st.session_state.conversation_id = st.session_state.conversation_id or str(__import__("uuid").uuid4())

    else:

        with st.spinner("Thinking…"):
            request  = ChatRequest(
                query           = query,
                conversation_id = st.session_state.conversation_id,
            )
            response = pipeline.chat(request, user_id=user_id)

        st.session_state.conversation_id = response.conversation_id
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    response.answer,
            "sources":    response.sources,
            "latency_ms": response.latency_ms,
        })
        st.session_state.last_response = {
            "query":      query,
            "answer":     response.answer,
            "latency_ms": response.latency_ms,
        }



def main() -> None:
    init_state()

    if not st.session_state.authenticated:
        render_auth_page()
    else:
        render_sidebar()
        render_main()


if __name__ == "__main__":
    main()