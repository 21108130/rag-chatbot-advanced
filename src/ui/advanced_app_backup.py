

import os
import sys
import time
import uuid
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
    page_title="RAG Chatbot v3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.app-title { font-family:'IBM Plex Mono',monospace; font-size:1.55rem; font-weight:600; color:#0F172A; letter-spacing:-0.5px; }
.app-subtitle { color:#64748B; font-size:0.83rem; margin-top:3px; }
.pill { display:inline-flex; align-items:center; gap:4px; padding:3px 10px; border-radius:20px; font-size:0.73rem; font-weight:600; margin:0 2px; }
.pill-blue   { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; }
.pill-green  { background:#F0FDF4; color:#15803D; border:1px solid #BBF7D0; }
.pill-purple { background:#FAF5FF; color:#7C3AED; border:1px solid #DDD6FE; }
.pill-amber  { background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; }
.pill-red    { background:#FFF1F2; color:#BE123C; border:1px solid #FECDD3; }
.pill-gray   { background:#F8FAFC; color:#475569; border:1px solid #CBD5E1; }
.source-card { background:#FFFDF0; border:1px solid #FDE68A; border-left:3px solid #F59E0B; border-radius:6px; padding:10px 14px; margin:6px 0; font-size:0.82rem; }
.source-header { font-weight:600; color:#92400E; margin-bottom:5px; }
.source-preview { color:#78716C; line-height:1.5; }
.score-bar-wrap { display:flex; align-items:center; gap:8px; margin:4px 0; }
.score-bar-bg { flex:1; background:#FEF3C7; border-radius:4px; height:6px; }
.score-bar-fill { height:6px; border-radius:4px; background:linear-gradient(90deg,#F59E0B,#10B981); }
.score-val { font-family:'IBM Plex Mono',monospace; font-size:0.73rem; color:#92400E; min-width:36px; }
.conf-badge { display:inline-flex; align-items:center; gap:4px; font-size:0.72rem; font-family:'IBM Plex Mono',monospace; padding:2px 8px; border-radius:4px; margin:0 3px; }
.conf-high   { background:#DCFCE7; color:#15803D; }
.conf-medium { background:#FEF9C3; color:#854D0E; }
.conf-low    { background:#FFE4E6; color:#BE123C; }
.provider-row { display:flex; align-items:center; justify-content:space-between; padding:8px 12px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; margin:4px 0; }
.provider-name { font-weight:600; font-size:0.85rem; color:#1E293B; }
.provider-stat { font-family:'IBM Plex Mono',monospace; font-size:0.73rem; color:#64748B; }
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; }
.dot-green { background:#22C55E; }
.dot-red   { background:#EF4444; }
.action-card { border-left:4px solid #EF4444; background:#FFF1F2; border-radius:0 8px 8px 0; padding:12px 16px; margin:6px 0; }
.action-card.week  { border-color:#F59E0B; background:#FFFBEB; }
.action-card.month { border-color:#22C55E; background:#F0FDF4; }
.action-rank { font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:0.95rem; }
.action-issue { font-size:0.84rem; color:#1E293B; margin:3px 0; font-weight:500; }
.action-fix { font-size:0.78rem; color:#64748B; font-style:italic; }
.doc-item { display:flex; align-items:flex-start; gap:8px; padding:7px 10px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; margin:4px 0; font-size:0.82rem; }
.doc-name { font-weight:600; color:#1E293B; }
.doc-meta { color:#94A3B8; font-size:0.73rem; }
.fb-bar { display:flex; align-items:center; gap:8px; padding:4px 0; border-top:1px solid #F1F5F9; margin-top:6px; }
</style>
""", unsafe_allow_html=True)


_PIPELINE_CACHE: dict = {}

def get_pipeline() -> AdvancedRAGPipeline:
    user_id = st.session_state.get("user_id", "anon")
    if user_id not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[user_id] = AdvancedRAGPipeline(
            enable_reranking=True, enable_streaming=True, max_context_tokens=3000,
        )
    st.session_state.pipeline = _PIPELINE_CACHE[user_id]
    return _PIPELINE_CACHE[user_id]


def init_state():
    defaults = {
        "authenticated": False, "user_id": None, "username": None,
        "pipeline": None, "messages": [], "conversation_id": None,
        "indexed_docs": [], "agent_mode": False, "use_streaming": True,
        "self_correct": False, "query_transform": False,
        "thumbs_down_msgid": None,
        "api_key_mode": "env",
        "user_provider_choice": "Groq",
        "user_groq_key": "", "user_gemini_key": "", "user_openrouter_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_auth_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 24px;">
            <div style="font-size:2.5rem;">🧠</div>
            <div class="app-title" style="font-size:1.8rem;text-align:center;">RAG Chatbot v3</div>
            <div class="app-subtitle" style="text-align:center;margin-top:6px;">
                Hybrid Search · Reranking · Streaming · Agent Mode · Self-Corrective RAG
            </div>
        </div>
        """, unsafe_allow_html=True)

        auth = get_auth_manager()
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Create Account"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                sub = st.form_submit_button("Login →", type="primary", use_container_width=True)
            if sub:
                if username and password:
                    user = auth.authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user.id
                        st.session_state.username = user.username
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                else:
                    st.warning("Please fill in both fields.")

        with tab_register:
            with st.form("reg_form"):
                ru = st.text_input("Username")
                re = st.text_input("Email")
                rp = st.text_input("Password", type="password")
                rs = st.form_submit_button("Create Account →", type="primary", use_container_width=True)
            if rs:
                if ru and re and rp:
                    try:
                        auth.create_user(ru, re, rp)
                        st.success(f"✅ Account created! Please login as **{ru}**.")
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.warning("Please fill in all fields.")


def render_sidebar():
    with st.sidebar:


        uname = st.session_state.username or "?"
        initial = uname[0].upper()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:6px 0;margin-bottom:4px;">
            <div style="width:32px;height:32px;border-radius:50%;
                background:linear-gradient(135deg,#667eea,#764ba2);
                display:flex;align-items:center;justify-content:center;
                color:white;font-weight:700;font-size:0.9rem;">{initial}</div>
            <div>
                <div style="font-weight:600;font-size:0.9rem;color:#1E293B;">{uname}</div>
                <div style="font-size:0.7rem;color:#94A3B8;">Active session</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Logout", use_container_width=True):
            uid = st.session_state.get("user_id")
            if uid and uid in _PIPELINE_CACHE:
                del _PIPELINE_CACHE[uid]
            if uid:
                try:
                    from src.llm.multi_provider_client import clear_user_client
                    clear_user_client(uid)
                except Exception:
                    pass
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.divider()


        st.markdown("**🔌 Active LLM Provider**")
        try:
            from src.llm.multi_provider_client import get_llm_client
            _llm = get_llm_client(user_id=st.session_state.get("user_id"))
            provider = _llm.get_active_provider()
            icons  = {"groq": "⚡", "gemini": "🔵", "openrouter": "🔀"}
            colors = {"groq": "pill-green", "gemini": "pill-blue", "openrouter": "pill-purple"}
            key_src = "🔑 Your Key" if _llm.using_user_keys() else "🔒 Owner Key"
            st.markdown(
                f'<span class="pill {colors.get(provider,"pill-gray")}">'
                f'{icons.get(provider,"🤖")} {provider.upper()}</span>'
                f'&nbsp;<span class="pill pill-gray" style="font-size:0.68rem">{key_src}</span>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.caption("Provider unknown")

        st.divider()


        st.markdown("**🔑 API Key**")

        _mode_labels = [
            "🔒  App's built-in key  (owner / tester)",
            "🔑  Enter my own key  (external user)",
        ]
        _mode_values = {"🔒  App's built-in key  (owner / tester)": "env",
                        "🔑  Enter my own key  (external user)": "user"}
        _cur_idx = 0 if st.session_state.api_key_mode == "env" else 1

        selected_label = st.selectbox(
            "API key source",
            options=_mode_labels,
            index=_cur_idx,
            label_visibility="collapsed",
            key="sb_api_key_mode_sel",
        )
        selected_mode = _mode_values[selected_label]
        st.session_state.api_key_mode = selected_mode

        # ── Mode A ─────────────────────────────────────────────────────────────
        if selected_mode == "env":
            env_has_keys = bool(
                settings.groq_api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
            )
            if env_has_keys:
                st.success("✅ App keys loaded from .env — ready to use.")
            else:
                st.error("⚠️ No keys found in .env — add keys or switch to 'Enter my own key'.")


        else:
            st.caption("Pick a provider, paste your key, click **Save & Activate**. Stored in your browser session only — never written to disk.")

            provider_choice = st.selectbox(
                "Provider",
                options = ["Groq", "Gemini", "OpenRouter"],
                index   = ["Groq", "Gemini", "OpenRouter"].index(
                    st.session_state.user_provider_choice
                ),
                help = "Groq = fast & free  |  Gemini = Google  |  OpenRouter = many models",
                key  = "sb_provider_select",
            )
            st.session_state.user_provider_choice = provider_choice

            placeholders = {
                "Groq":       "gsk_…   get a free key at console.groq.com",
                "Gemini":     "AIza…   get a free key at aistudio.google.com",
                "OpenRouter": "sk-or-… get a free key at openrouter.ai",
            }
            saved_vals = {
                "Groq":       st.session_state.user_groq_key,
                "Gemini":     st.session_state.user_gemini_key,
                "OpenRouter": st.session_state.user_openrouter_key,
            }

            key_input = st.text_input(
                label       = f"{provider_choice} API Key",
                type        = "password",
                value       = saved_vals[provider_choice],
                placeholder = placeholders[provider_choice],
                key         = f"sb_keyinput_{provider_choice}",
            )

            col_save, col_clear = st.columns(2)
            with col_save:
                if st.button("💾 Save & Activate", type="primary",
                             use_container_width=True, key="sb_btn_save"):
                    if not key_input.strip():
                        st.warning("Paste your API key above first.")
                    else:
                        if provider_choice == "Groq":
                            st.session_state.user_groq_key = key_input.strip()
                        elif provider_choice == "Gemini":
                            st.session_state.user_gemini_key = key_input.strip()
                        elif provider_choice == "OpenRouter":
                            st.session_state.user_openrouter_key = key_input.strip()
                        try:
                            from src.llm.multi_provider_client import get_llm_client
                            get_llm_client(
                                user_id           = st.session_state.get("user_id"),
                                groq_api_key      = st.session_state.user_groq_key or None,
                                gemini_api_key    = st.session_state.user_gemini_key or None,
                                openrouter_api_key= st.session_state.user_openrouter_key or None,
                            )
                            st.success(f"✅ {provider_choice} key active!")
                        except Exception as e:
                            st.error(f"Failed to apply key: {e}")
                        st.rerun()

            with col_clear:
                if st.button("🗑️ Clear", use_container_width=True, key="sb_btn_clear"):
                    st.session_state.user_groq_key       = ""
                    st.session_state.user_gemini_key     = ""
                    st.session_state.user_openrouter_key = ""
                    try:
                        from src.llm.multi_provider_client import clear_user_client
                        clear_user_client(st.session_state.get("user_id", ""))
                    except Exception:
                        pass
                    st.rerun()


            pills = []
            if st.session_state.user_groq_key:
                pills.append('<span class="pill pill-green">⚡ Groq ✓</span>')
            if st.session_state.user_gemini_key:
                pills.append('<span class="pill pill-blue">🔵 Gemini ✓</span>')
            if st.session_state.user_openrouter_key:
                pills.append('<span class="pill pill-purple">🔀 OpenRouter ✓</span>')
            if pills:
                st.markdown(" ".join(pills), unsafe_allow_html=True)
            else:
                st.warning("⚠️ No key saved yet.")

        st.divider()


        st.markdown("**⚙️ Settings**")
        st.session_state.use_streaming   = st.toggle("⚡ Streaming",       value=st.session_state.use_streaming,
                                                      help="Stream tokens word-by-word")
        st.session_state.agent_mode      = st.toggle("🤖 Agent Mode",      value=st.session_state.agent_mode,
                                                      help="Multi-step reasoning + web search")
        st.session_state.self_correct    = st.toggle("🔄 Self-Correction", value=st.session_state.self_correct,
                                                      help="Re-retrieves on low confidence (Enhancement 8)")
        st.session_state.query_transform = st.toggle("🔀 Multi-Query",     value=st.session_state.query_transform,
                                                      help="3 query variants for better recall (Enhancement 6)")

        st.divider()


        st.markdown("**📄 Upload Documents**")
        uploaded_files = st.file_uploader(
            "files", type=settings.allowed_extensions,
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if uploaded_files:
            if st.button("🚀 Index Documents", type="primary", use_container_width=True):
                handle_upload(uploaded_files)

        if st.session_state.indexed_docs:
            st.divider()
            st.markdown("**📚 Knowledge Base**")
            for doc in st.session_state.indexed_docs:
                icon  = "✅" if doc.status == DocumentStatus.INDEXED else "❌"
                q     = getattr(doc, "quality_score", None)
                q_str = f" · Q:{q:.0%}" if q else ""
                st.markdown(
                    f'<div class="doc-item"><span style="font-size:1rem">{icon}</span>'
                    f'<div><div class="doc-name">{doc.filename}</div>'
                    f'<div class="doc-meta">{doc.chunk_count} chunks{q_str}</div></div></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        render_kb_stats()

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages          = []
            st.session_state.conversation_id   = None
            st.session_state.thumbs_down_msgid = None
            st.rerun()


def handle_upload(uploaded_files):
    pipeline    = get_pipeline()
    upload_path = settings.upload_path
    with st.spinner("Indexing documents…"):
        for uf in uploaded_files:
            if uf.size > settings.max_file_size_bytes:
                st.sidebar.error(f"❌ '{uf.name}' exceeds {settings.max_file_size_mb}MB.")
                continue
            dest = upload_path / uf.name
            dest.write_bytes(uf.getbuffer())
            quality_score = None
            try:
                from src.extraction.quality_extractor import QualityExtractor
                qr = QualityExtractor().extract(dest)
                quality_score = qr.quality_score
                if not qr.is_acceptable:
                    st.sidebar.warning(f"⚠️ **{uf.name}** low quality ({quality_score:.0%}): " + ", ".join(qr.issues[:2]))
            except Exception:
                pass
            result = pipeline.ingest_document(dest, user_id=st.session_state.user_id)
            if quality_score is not None:
                result.__dict__["quality_score"] = quality_score
            st.session_state.indexed_docs.append(result)
            if result.status == DocumentStatus.INDEXED:
                q_info = f" · quality {quality_score:.0%}" if quality_score else ""
                st.sidebar.success(f"✅ **{uf.name}** — {result.chunk_count} chunks{q_info}")
            else:
                st.sidebar.error(f"❌ {uf.name}: {result.message}")


def render_kb_stats():
    pipeline = get_pipeline()
    try:
        stats = pipeline.get_kb_stats(user_id=st.session_state.user_id)
        col1, col2 = st.columns(2)
        col1.metric("Documents", stats["total_docs"])
        col2.metric("Chunks",    stats["total_chunks"])
    except Exception:
        st.caption("Stats unavailable")


def render_main():
    col_title, col_pills = st.columns([3, 5])
    with col_title:
        st.markdown(
            '<div class="app-title">🧠 RAG Chatbot <span style="color:#94A3B8;font-weight:300">v3</span></div>'
            '<div class="app-subtitle">Hybrid Search · Reranking · Self-Corrective · Agent Mode</div>',
            unsafe_allow_html=True,
        )
    with col_pills:
        pills = '<span class="pill pill-blue">🔀 Hybrid</span>'
        pills += '<span class="pill pill-green">🎯 Reranking</span>'
        if st.session_state.use_streaming:    pills += '<span class="pill pill-purple">⚡ Streaming</span>'
        if st.session_state.agent_mode:       pills += '<span class="pill pill-amber">🤖 Agent</span>'
        if st.session_state.self_correct:     pills += '<span class="pill pill-green">🔄 Self-Correct</span>'
        if st.session_state.query_transform:  pills += '<span class="pill pill-blue">🔀 Multi-Query</span>'
        st.markdown(f'<div style="padding-top:14px;">{pills}</div>', unsafe_allow_html=True)

    st.divider()
    tab_chat, tab_analytics, tab_eval, tab_routing = st.tabs([
        "💬 Chat", "📊 Analytics", "🧪 Evaluation", "🔌 Routing & Health",
    ])
    with tab_chat:      render_chat_tab()
    with tab_analytics: render_analytics_tab()
    with tab_eval:      render_eval_tab()
    with tab_routing:   render_routing_tab()


def render_chat_tab():
    if not st.session_state.messages:
        st.info("👈 **Upload a document** in the sidebar, then ask questions here.\n\nToggle **🤖 Agent Mode** in the sidebar for web search + multi-step reasoning.", icon="💡")

    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message("user" if role == "user" else "assistant", avatar="🧑" if role == "user" else "🧠"):
            st.markdown(msg["content"])
            if role == "assistant":
                conf = msg.get("confidence")
                corr = msg.get("correction_applied", "none")
                if conf is not None:
                    level = "high" if conf >= 0.55 else "medium" if conf >= 0.30 else "low"
                    badge = f'<span class="conf-badge conf-{level}">🎯 confidence: {conf:.2f}</span>'
                    if corr and corr != "none":
                        badge += f' <span class="conf-badge conf-medium">🔄 {corr}</span>'
                    st.markdown(badge, unsafe_allow_html=True)

                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📎 {len(sources)} source(s) retrieved", expanded=False):
                        for i, src in enumerate(sources, 1):
                            fname    = src.metadata.get("source", src.doc_id) if hasattr(src, "metadata") else "Unknown"
                            score    = src.similarity_score if hasattr(src, "similarity_score") else 0.0
                            reranked = src.metadata.get("reranked", False) if hasattr(src, "metadata") else False
                            method   = src.metadata.get("retrieval_method", "vector") if hasattr(src, "metadata") else "vector"
                            preview  = (src.content[:280] if hasattr(src, "content") else "").replace("\n", " ").strip()
                            pct      = int(min(score * 100, 100))
                            rerank_tag = ' <span class="pill pill-green" style="font-size:0.65rem;">✓ reranked</span>' if reranked else ""
                            method_tag = f' <span class="pill pill-gray" style="font-size:0.65rem;">{method}</span>'
                            st.markdown(
                                f'<div class="source-card"><div class="source-header">Source {i}: {fname}{rerank_tag}{method_tag}</div>'
                                f'<div class="score-bar-wrap"><span style="font-size:0.7rem;color:#92400E;">Relevance</span>'
                                f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
                                f'<span class="score-val">{score:.3f}</span></div>'
                                f'<div class="source-preview">{preview}…</div></div>',
                                unsafe_allow_html=True,
                            )

                agent_steps = msg.get("agent_steps", [])
                if agent_steps:
                    with st.expander(f"🔍 Agent reasoning — {len(agent_steps)} step(s)", expanded=False):
                        for step in agent_steps:
                            st.markdown(f"**Step {step['iteration']}** → `{step['tool']}` with `{step.get('args', {})}`\n\n> {step.get('observation', '')[:300]}")

                lat    = msg.get("latency_ms")
                msg_id = msg.get("msg_id", "")
                rated  = msg.get("feedback")
                fcol1, fcol2, fcol3 = st.columns([1, 1, 10])
                with fcol1:
                    if rated is None:
                        if st.button("👍", key=f"up_{msg_id}", help="Good answer"):
                            _record_feedback(msg, 1); st.rerun()
                    elif rated == 1:
                        st.caption("👍")
                with fcol2:
                    if rated is None:
                        if st.button("👎", key=f"dn_{msg_id}", help="Poor answer"):
                            st.session_state.thumbs_down_msgid = msg_id; st.rerun()
                    elif rated == -1:
                        st.caption("👎")
                with fcol3:
                    parts = []
                    if lat: parts.append(f"⏱ {lat:.0f}ms")
                    if msg.get("failure_type"): parts.append(f"❗ {msg['failure_type']}")
                    if parts: st.caption(" · ".join(parts))

    if st.session_state.thumbs_down_msgid:
        render_feedback_dialog(st.session_state.thumbs_down_msgid)

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        placeholder = "Ask anything — I'll search your docs + the web!" if st.session_state.agent_mode else "Ask a question about your documents…"
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            user_input = st.text_input("Message", placeholder=placeholder, label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("Send →", type="primary", use_container_width=True)
    if submitted and user_input.strip():
        handle_chat(user_input.strip()); st.rerun()


def render_feedback_dialog(msg_id: str):
    with st.container():
        st.markdown("---")
        st.markdown("#### 👎 What went wrong? ")
        FAILURE_OPTIONS = {
            "🔴 Hallucination": "hallucination", "🔵 Retrieval Gap": "retrieval_gap",
            "🟠 Partial Answer": "partial_answer", "🟡 Missing Citation": "citation_missing",
            "⚪ Off-Topic": "off_topic", "🟣 Wrong Format": "wrong_format", "❔ Other": "other",
        }
        col_type, col_text = st.columns([2, 3])
        with col_type:
            failure_label = st.selectbox("Failure type", list(FAILURE_OPTIONS.keys()), key="fb_sel")
        with col_text:
            correction = st.text_input("Correct answer (optional)", placeholder="What should the answer have said?", key="fb_correction")
        col_sub, col_skip = st.columns([1, 1])
        with col_sub:
            if st.button("✅ Submit Feedback", type="primary", use_container_width=True):
                for msg in st.session_state.messages:
                    if msg.get("msg_id") == msg_id:
                        _record_structured_feedback(msg, failure_type=FAILURE_OPTIONS[failure_label], expected=correction.strip() if correction.strip() else None)
                        break
                st.session_state.thumbs_down_msgid = None
                st.success("Feedback recorded. 📊"); st.rerun()
        with col_skip:
            if st.button("Skip", use_container_width=True):
                for msg in st.session_state.messages:
                    if msg.get("msg_id") == msg_id:
                        _record_feedback(msg, -1); break
                st.session_state.thumbs_down_msgid = None; st.rerun()


def _record_feedback(msg: dict, rating: int):
    try:
        get_pipeline().record_feedback(
            user_id=st.session_state.user_id, query=msg.get("query", ""),
            answer=msg.get("content", ""), rating=rating,
            conversation_id=st.session_state.conversation_id, latency_ms=msg.get("latency_ms"),
        )
        msg["feedback"] = rating
    except Exception as e:
        logger.warning(f"Feedback error: {e}")


def _record_structured_feedback(msg: dict, failure_type: str, expected: str = None):
    try:
        from src.feedback.structured_feedback import StructuredFeedbackCollector, FailureType
        from src.prioritization.priority_framework import get_priority_framework
        collector = StructuredFeedbackCollector()
        ft = FailureType(failure_type)
        collector.record(
            user_id=st.session_state.user_id, query=msg.get("query", ""),
            answer=msg.get("content", ""), rating=-1, failure_type=ft,
            expected_response=expected,
            retrieved_doc_names=[s.metadata.get("source", s.doc_id) for s in msg.get("sources", []) if hasattr(s, "metadata")],
            retrieval_scores=[s.similarity_score for s in msg.get("sources", []) if hasattr(s, "similarity_score")],
            confidence_scores={"retrieval": msg.get("confidence", 0.0)} if msg.get("confidence") else {},
            latency_ms=msg.get("latency_ms"),
        )
        get_priority_framework().record_query_failure(query=msg.get("query", ""), failure_type=failure_type, user_id=st.session_state.user_id)
        msg["feedback"] = -1
        msg["failure_type"] = failure_type
    except Exception as e:
        logger.warning(f"Structured feedback error: {e}")
        _record_feedback(msg, -1)


def handle_chat(query: str):
    pipeline = get_pipeline()
    user_id  = st.session_state.user_id
    msg_id   = str(uuid.uuid4())[:8]
    st.session_state.messages.append({"role": "user", "content": query, "msg_id": msg_id + "_u", "query": query})

    if st.session_state.agent_mode:
        try:
            from src.agents.rag_agent import RAGAgent
            stats    = pipeline.get_kb_stats(user_id=user_id)
            has_docs = stats.get("total_chunks", 0) > 0
            agent    = RAGAgent(pipeline=pipeline, user_id=user_id, has_docs=has_docs)
            with st.spinner("🤖 Agent reasoning…"):
                result = agent.run(query)
            answer      = result["answer"]
            agent_steps = result["steps"]
        except Exception as exc:
            answer = f"Agent error: {exc}"; agent_steps = []
        st.session_state.messages.append({"role": "assistant", "content": answer, "agent_steps": agent_steps, "msg_id": msg_id, "query": query})
        return

    try:
        total_chunks = pipeline.get_kb_stats(user_id=user_id).get("total_chunks", 0)
    except Exception:
        total_chunks = 0
    if total_chunks == 0:
        st.session_state.messages.pop()
        st.warning("⚠️ No documents in knowledge base. Upload a document first.")
        return

    if st.session_state.self_correct:
        try:
            from src.self_correct.reflective_rag import CorrectiveRAG
            corrective = CorrectiveRAG(pipeline=pipeline)
            with st.spinner("🔄 Checking retrieval confidence…"):
                result = corrective.chat_with_correction(query, user_id=user_id)
            confidence = result["confidence"].overall
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result.get("sources", []), "confidence": confidence, "correction_applied": result.get("correction_applied", "none"), "latency_ms": result.get("latency_ms"), "msg_id": msg_id, "query": query})
            return
        except Exception as exc:
            logger.warning(f"Self-correct failed: {exc}")

    if st.session_state.query_transform:
        try:
            from src.query_transform.transformer import QueryTransformer
            transformer = QueryTransformer()
            variants    = transformer.transform(query, strategy="multi_query", n=3)
            retriever   = pipeline._get_retriever(user_id)
            all_chunks  = []
            for v in variants:
                r = retriever.retrieve(query=v, top_k=4)
                all_chunks.extend(r.chunks)
            deduped   = transformer.deduplicate_results(all_chunks)[:5]
            from src.utils.models import RetrievalResult
            merged    = RetrievalResult(query=query, chunks=deduped)
            optimized = pipeline._optimizer.optimize(merged, query=query)
            context   = pipeline._optimizer.to_context_string(optimized)
            llm_res   = pipeline.llm.generate(query=query, context=context)
            st.session_state.messages.append({"role": "assistant", "content": llm_res["answer"], "sources": deduped, "latency_ms": None, "msg_id": msg_id, "query": query})
            return
        except Exception as exc:
            logger.warning(f"Query transform failed: {exc}")

    if st.session_state.use_streaming:
        with st.chat_message("assistant", avatar="🧠"):
            placeholder = st.empty()
            full_answer = ""
            t0 = time.perf_counter()
            for token in pipeline.chat_stream(query=query, user_id=user_id, conversation_id=st.session_state.conversation_id):
                full_answer += token
                placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)
            latency_ms = (time.perf_counter() - t0) * 1000
            st.caption(f"⏱ {latency_ms:.0f}ms · ⚡ streamed")
        st.session_state.messages.append({"role": "assistant", "content": full_answer, "latency_ms": latency_ms, "msg_id": msg_id, "query": query})
        st.session_state.conversation_id = st.session_state.conversation_id or str(uuid.uuid4())
    else:
        with st.spinner("Thinking…"):
            response = pipeline.chat(ChatRequest(query=query, conversation_id=st.session_state.conversation_id), user_id=user_id)
        st.session_state.conversation_id = response.conversation_id
        st.session_state.messages.append({"role": "assistant", "content": response.answer, "sources": response.sources, "latency_ms": response.latency_ms, "msg_id": msg_id, "query": query})


def render_analytics_tab():
    pipeline = get_pipeline()
    st.markdown("### 📈 Pipeline Performance")
    try:
        metrics  = pipeline.get_metrics_summary()
        feedback = pipeline.get_feedback_summary(user_id=st.session_state.user_id)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", metrics["total_queries"])
        c2.metric("Avg Latency",   f"{metrics['avg_latency_ms']:.0f} ms")
        c3.metric("P95 Latency",   f"{metrics['p95_latency_ms']:.0f} ms")
        c4.metric("Error Rate",    f"{metrics['error_rate']:.1f}%")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Tokens",  metrics["total_tokens"])
        c6.metric("Avg Tokens",    f"{metrics['avg_tokens']:.0f}")
        c7.metric("Satisfaction",  f"{feedback.get('satisfaction_rate', 0):.0f}%")
        c8.metric("Prometheus",    "✅ ON" if metrics["prometheus_enabled"] else "❌ OFF")
    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")

    st.divider()
    st.markdown("### 🔴 Failure Analysis ")
    try:
        from src.feedback.structured_feedback import get_structured_feedback
        analysis = get_structured_feedback().get_failure_analysis(user_id=st.session_state.user_id)
        if analysis["total"] == 0:
            st.info("No feedback yet. Click 👎 on any answer to track failures.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Total Feedback", analysis["total"])
            a2.metric("Positive 👍",    analysis["positive"])
            a3.metric("Negative 👎",    analysis["negative"])
            if analysis.get("top_issues"):
                st.markdown("**Top Failure Types:**")
                for issue_type, count in analysis["top_issues"]:
                    pct = round(count / max(analysis["negative"], 1) * 100)
                    st.progress(pct / 100, text=f"`{issue_type}` — {count}x ({pct}%)")
            if analysis.get("recommended_fixes"):
                st.markdown("**Recommended Fixes:**")
                for fix in analysis["recommended_fixes"][:3]:
                    st.markdown(f'<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:6px;padding:8px 12px;margin:4px 0;font-size:0.84rem;"><strong>{fix["label"]}</strong> ({fix["count"]}x)<br><span style="color:#64748B;">💡 {fix["fix"]}</span></div>', unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Structured feedback: {e}")

    st.divider()
    st.markdown("### 🎯 Priority Action Plan ")
    try:
        from src.prioritization.priority_framework import get_priority_framework
        actions = get_priority_framework().get_action_plan(top_n=5)
        if not actions:
            st.info("No failures recorded yet.")
        else:
            for a in actions:
                css   = {"immediate": "action-card", "this_week": "action-card week", "this_month": "action-card month"}.get(a.urgency, "action-card")
                emoji = {"immediate": "🔴", "this_week": "🟡", "this_month": "🟢"}.get(a.urgency, "⚪")
                st.markdown(f'<div class="{css}"><div class="action-rank">#{a.rank} {emoji} {a.urgency.upper()} · score: {a.priority_score:.1f}</div><div class="action-issue">{a.issue}</div><div class="action-fix">💡 {a.suggested_fix}</div></div>', unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Priority framework: {e}")

    st.divider()
    st.markdown("### 🔍 Recent Low-Rated Queries")
    try:
        feedback = pipeline.get_feedback_summary(user_id=st.session_state.user_id)
        bad = feedback.get("bad_queries", [])
        if bad:
            for bq in bad[:8]:
                comment = f'<br><span style="color:#64748B;font-style:italic">{bq["comment"]}</span>' if bq.get("comment") else ""
                st.markdown(f'<div style="padding:7px 12px;background:#FFF1F2;border-left:3px solid #EF4444;border-radius:0 6px 6px 0;margin:4px 0;font-size:0.82rem;"><code>{bq["query"]}</code>{comment}</div>', unsafe_allow_html=True)
        else:
            st.info("No negative feedback yet.")
    except Exception:
        st.info("Feedback data unavailable.")


def render_eval_tab():
    st.markdown("### 🧪 Automated Evaluation ")
    st.markdown("Run the built-in 50-query test suite after any config change. Scores: **Retrieval Precision**, **Answer Correctness (F1)**, **Groundedness**.")
    ca, cb, cc = st.columns(3)
    eval_cat = ca.selectbox("Category", ["all", "factual", "reasoning", "summary", "comparison", "multi-hop"])
    eval_pri = cb.selectbox("Priority",  ["all", "high", "medium", "low"])
    max_q    = cc.number_input("Max queries", min_value=1, max_value=50, value=10)
    if st.button("▶️ Run Evaluation", type="primary"):
        try:
            from src.evaluation.eval_pipeline import EvaluationPipeline
            from src.evaluation.eval_dataset import load_eval_dataset
            queries = load_eval_dataset(category=None if eval_cat == "all" else eval_cat, priority=None if eval_pri == "all" else eval_pri, max_queries=int(max_q))
            if not queries:
                st.warning("No queries match those filters."); return
            progress  = st.progress(0, text="Starting evaluation…")
            evaluator = EvaluationPipeline(get_pipeline())
            results   = []
            for i, q in enumerate(queries):
                progress.progress((i + 1) / len(queries), text=f"Running {q.query_id} ({i+1}/{len(queries)})…")
                results.append(evaluator._evaluate_one(q))
            progress.empty()
            n = len(results); valid = [r for r in results if r.error is None]; passed = [r for r in valid if r.passed]
            def avg(v): return round(sum(v) / len(v), 3) if v else 0.0
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Pass Rate",    f"{round(len(passed)/max(n,1)*100,1)}%", delta=f"{len(passed)}/{n}")
            s2.metric("Retrieval",    avg([r.retrieval_precision for r in valid]))
            s3.metric("Correctness",  avg([r.answer_correctness for r in valid]))
            s4.metric("Groundedness", avg([r.groundedness for r in valid]))
            s5.metric("Avg Latency",  f"{avg([r.latency_ms for r in valid]):.0f}ms")
            st.divider(); st.markdown("**Per-Query Results:**")
            for r in results:
                cols = st.columns([2, 5, 1, 1, 1, 1, 1])
                cols[0].caption(r.query_id); cols[1].caption(r.query[:55] + ("…" if len(r.query) > 55 else ""))
                cols[2].caption(f"{r.retrieval_precision:.2f}"); cols[3].caption(f"{r.answer_correctness:.2f}")
                cols[4].caption(f"{r.groundedness:.2f}"); cols[5].caption(f"{r.overall_score:.2f}")
                cols[6].caption("❌ ERR" if r.error else ("✅" if r.passed else "❌"))
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            import traceback; st.code(traceback.format_exc())


def render_routing_tab():
    st.markdown("### 🔌 LLM Provider Health *( Smart Routing)*")
    try:
        from src.routing.smart_router import get_smart_router, RoutingStrategy
        router = get_smart_router(); status = router.get_status()
        for pname, s in status.items():
            dot  = "dot-green" if s["status"] == "CLOSED" else "dot-red"
            slbl = "🟢 HEALTHY" if s["status"] == "CLOSED" else "🔴 CIRCUIT OPEN"
            st.markdown(f'<div class="provider-row"><div><span class="status-dot {dot}"></span><span class="provider-name">{pname.upper()}</span>&nbsp;<span class="pill pill-gray" style="font-size:0.68rem">{slbl}</span></div><div style="display:flex;gap:18px;"><span class="provider-stat">⏱ {s["avg_latency_ms"]:.0f}ms avg</span><span class="provider-stat">✓ {s["success_count"]} ok</span><span class="provider-stat">✗ {s["total_errors"]} err</span><span class="provider-stat">📉 {s["error_rate"]:.1f}% err rate</span><span class="provider-stat">Priority: {s["priority"]}</span></div></div>', unsafe_allow_html=True)
        st.markdown("**Manual Circuit Reset:**")
        col_prov, col_btn, col_all = st.columns([3, 2, 2])
        with col_prov:  reset_target = st.selectbox("Provider", list(status.keys()), key="reset_sel")
        with col_btn:
            if st.button("🔄 Reset Selected", use_container_width=True):
                router.reset_circuit(reset_target); st.success(f"Circuit reset for {reset_target}"); st.rerun()
        with col_all:
            if st.button("🔄 Reset All", use_container_width=True):
                router.reset_all_circuits(); st.success("All circuits reset"); st.rerun()
        st.divider(); st.markdown("**🧪 Test a Routing Strategy:**")
        strategy_map = {"Priority (best first)": RoutingStrategy.PRIORITY, "Cost-Aware (cheapest first)": RoutingStrategy.COST_AWARE, "Load Balance (round-robin)": RoutingStrategy.LOAD_BALANCE, "Fastest (by latency history)": RoutingStrategy.FASTEST}
        chosen = st.selectbox("Strategy", list(strategy_map.keys()))
        test_q = st.text_input("Test query", value="What is retrieval augmented generation?")
        if st.button("🚀 Test Route", type="primary"):
            with st.spinner(f"Routing via {chosen}…"):
                try:
                    result = router.route(messages=[{"role": "user", "content": test_q}], strategy=strategy_map[chosen], max_tokens=200)
                    st.success(f"✅ **{result.get('provider','?').upper()}** responded in **{result.get('latency_ms', 0):.0f}ms** ({result.get('tokens_used', 0)} tokens)")
                    st.markdown(result["answer"][:500])
                except Exception as e:
                    st.error(f"Routing failed: {e}")
    except Exception as e:
        st.info(f"Smart router not available: {e}")

    st.divider()
    st.markdown("### 🔐 Tenant Isolation")
    try:
        from src.tenant.isolation import get_tenant_manager, IsolationLevel
        tm = get_tenant_manager(); tenants = tm.list_tenants()
        if tenants:
            level_color = {"high": "pill-green", "medium": "pill-blue", "low": "pill-gray"}
            for t in tenants:
                st.markdown(f'<div class="provider-row"><span class="provider-name">{t["tenant_id"]}</span><div style="display:flex;gap:10px;align-items:center;"><span class="pill {level_color.get(t["isolation_level"],"pill-gray")}">{t["isolation_level"].upper()}</span><span class="provider-stat">plan: {t["plan"]}</span><span class="provider-stat">{t["doc_count"]} docs · {t["query_count"]} queries</span><span class="provider-stat">compliance: {", ".join(t["compliance"])}</span></div></div>', unsafe_allow_html=True)
        else:
            st.info("Current user uses **MEDIUM isolation** (separate ChromaDB collection).\n\nRegister tenants below to assign custom isolation levels.")
        with st.expander("➕ Register Tenant"):
            ct1, ct2, ct3 = st.columns(3)
            new_tid  = ct1.text_input("Tenant ID")
            new_lv   = ct2.selectbox("Isolation Level", ["medium", "high", "low"])
            new_plan = ct3.selectbox("Plan", ["standard", "premium", "enterprise"])
            if st.button("Register", type="primary"):
                if new_tid:
                    tm.register_tenant(new_tid, IsolationLevel(new_lv), plan=new_plan)
                    st.success(f"✅ Registered: {new_tid} ({new_lv} isolation)"); st.rerun()
    except Exception as e:
        st.info(f"Tenant manager: {e}")


def main():
    init_state()
    if not st.session_state.authenticated:
        render_auth_page()
    else:
        render_sidebar()
        render_main()


if __name__ == "__main__":
    main()