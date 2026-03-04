# 🧠 Advanced RAG Chatbot v3

> **Production-grade Retrieval-Augmented Generation system** with hybrid search, self-corrective retrieval, multi-provider LLM routing, agentic reasoning, real-time streaming, and full multi-tenant support — built entirely in Python without LangChain or LlamaIndex.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-green.svg)](https://chromadb.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 What Makes This Different

Most RAG chatbots on the internet are tutorials: one PDF, one LLM call, a few hundred lines of code. This project is a **full production system** with features you would only find in enterprise products:

| Feature | Typical Tutorial | This Project |
| --- | --- | --- |
| Search method | Vector only | **Hybrid: Vector + BM25 + RRF fusion** |
| Retrieval quality | No reranking | **Cross-encoder reranking (ms-marco)** |
| LLM providers | 1 hardcoded | **3 providers with circuit breakers + auto-fallback** |
| Hallucination control | None | **Self-corrective loop with confidence scoring** |
| Chunking | Fixed-size | **4 strategies: semantic, overlap, dynamic, sentence-window** |
| Query handling | Raw input | **Multi-query generation + HyDE + query expansion** |
| Multi-user | No | **JWT auth + per-user isolated vector stores** |
| Tenant isolation | No | **3 levels: HIGH / MEDIUM / LOW (HIPAA-ready)** |
| Feedback system | Thumbs up/down | **Structured failure categories + Pareto analysis** |
| Observability | None | **Prometheus metrics + Grafana dashboards** |
| Evaluation | Manual | **Automated 50-query test suite with F1 scoring** |
| Deployment | `python app.py` | **Docker Compose + Nginx + FastAPI REST API** |
| Agent mode | No | **ReAct loop: KB search + live web search (DuckDuckGo)** |
| API keys | One .env | **Dual mode: owner keys OR user-supplied keys** |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[JWT Auth]              Per-user session, role management
    │
    ▼
[Query Transform]       Multi-query generation / HyDE / expansion
    │
    ▼
[Hybrid Retriever]      Vector (ChromaDB) + BM25 → Reciprocal Rank Fusion
    │
    ▼
[Cross-Encoder Reranker] ms-marco-MiniLM-L-6-v2 — reorders by true relevance
    │
    ▼
[Self-Corrective Loop]  Confidence scoring → re-retrieve if score < 0.55
    │
    ▼
[Context Optimizer]     Token counting, dedup, smart truncation (3000 tokens)
    │
    ▼
[Multi-Provider LLM]    Groq → Gemini → OpenRouter (circuit breaker failover)
    │
    ▼
[Prometheus Metrics]    Latency histograms, token counters, quality gauges
    │
    ▼
[Structured Feedback]   Failure categorisation → priority action plan
    │
    ▼
ChatResponse            Answer + sources + confidence + latency
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/rag-chatbot-v3.git
cd rag-chatbot-v3
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp  .env
```

Edit `.env`:

```env
# Required — at least one LLM provider
GROQ_API_KEY=gsk_...          # free at console.groq.com
GEMINI_API_KEY=AIza...         # free at aistudio.google.com
OPENROUTER_API_KEY=sk-or-...   # free at openrouter.ai

# Optional overrides
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RESULTS=5
CHUNK_SIZE=512
```

### 3. Run

```bash
# Streamlit 
streamlit run src/ui/advanced_app.py
# Opens at http://localhost:8501

# FastAPI REST
uvicorn src.api.fastapi_app:app --port 8000 --reload
# Docs at http://localhost:8000/docs
```

### 4. Docker (Full Stack)

```bash
docker-compose up -d
```

Starts: Streamlit UI · FastAPI · Prometheus · Grafana · Nginx reverse proxy

---

## 📁 Project Structure

```
rag-chatbot-v3/
│
├── config/
│   └── settings.py                  # Pydantic-settings central config
│
├── src/
│   ├── agents/
│   │   └── rag_agent.py             # ReAct agentic loop (no LangChain)
│   │
│   ├── api/
│   │   └── fastapi_app.py           # REST API with rate limiting (SlowAPI)
│   │
│   ├── auth/
│   │   ├── auth_manager.py          # JWT + bcrypt authentication
│   │   └── models.py                # SQLAlchemy User/APIKey models
│   │
│   ├── chunking/
│   │   └── advanced_chunker.py      # 4 chunking strategies
│   │
│   ├── embeddings/
│   │   └── embedder.py              # Sentence-transformers (all-MiniLM-L6-v2)
│   │
│   ├── evaluation/
│   │   ├── eval_pipeline.py         # Automated evaluation loop
│   │   └── eval_dataset.py          # 50 built-in test queries
│   │
│   ├── extraction/
│   │   └── quality_extractor.py     # Document quality scoring + cleaning
│   │
│   ├── feedback/
│   │   ├── feedback_store.py        # Basic ratings store (SQLite)
│   │   └── structured_feedback.py  # Failure categorisation + Pareto
│   │
│   ├── llm/
│   │   ├── groq_client.py           # Groq LLM wrapper
│   │   ├── groq_streaming_client.py # Token streaming
│   │   └── multi_provider_client.py # 3-provider failover client
│   │
│   ├── observability/
│   │   ├── metrics.py               # Prometheus metrics collector
│   │   └── rag_metrics.py           # RAG-specific quality metrics
│   │
│   ├── prioritization/
│   │   └── priority_framework.py    # Weighted failure priority scoring
│   │
│   ├── query_transform/
│   │   └── transformer.py           # Multi-query, HyDE, expansion
│   │
│   ├── retrieval/
│   │   ├── advanced_rag_pipeline.py # Main pipeline orchestrator
│   │   ├── context_optimizer.py     # Token budget management
│   │   ├── hybrid_retriever.py      # Vector + BM25 + RRF
│   │   ├── indexer.py               # Document ingestion + chunking
│   │   └── reranker.py              # Cross-encoder reranker
│   │
│   ├── routing/
│   │   └── smart_router.py          # Circuit breaker + 4 routing strategies
│   │
│   ├── self_correct/
│   │   └── reflective_rag.py        # Self-corrective retrieval loop
│   │
│   ├── tenant/
│   │   └── isolation.py             # 3-level multi-tenant isolation
│   │
│   ├── ui/
│   │   └── advanced_app.py          # Streamlit 4-tab interface
│   │
│   ├── utils/
│   │   ├── document_loader.py       # PDF/DOCX/TXT/MD loader
│   │   ├── logger.py                # Loguru structured logging
│   │   ├── models.py                # Pydantic data models
│   │   └── text_splitter.py         # Basic text splitting utility
│   │
│   └── vectordb/
│       └── chroma_store.py          # ChromaDB vector store wrapper
│
├── tests/                           # pytest test suite
├── monitoring/
│   └── prometheus.yml               # Prometheus scrape config
├── nginx/
│   └── nginx.conf                   # Reverse proxy config
├── docker-compose.yml               # Full stack deployment
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🔬 Feature Deep-Dives

### 1. Hybrid Search with Reciprocal Rank Fusion

Standard RAG uses only vector (semantic) search. This system combines two complementary signals:

- **Dense retrieval** — `all-MiniLM-L6-v2` embeddings in ChromaDB (semantic meaning)
- **Sparse retrieval** — BM25Okapi keyword matching (exact term matching)

Results are merged using **Reciprocal Rank Fusion**:

```
RRF_score = Σ 1 / (k + rank_i)    where k = 60
```

This consistently outperforms either method alone, especially on queries containing specific names, numbers, or technical terms where pure vector search degrades.

**Weights:** `Vector: 0.7 · BM25: 0.3` (configurable)

---

### 2. Cross-Encoder Reranking

After hybrid retrieval returns top-20 candidates, a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) jointly encodes each `(query, passage)` pair and produces a direct relevance score.

Unlike bi-encoder cosine similarity which embeds query and passage separately, cross-encoders see both simultaneously — dramatically improving precision. The top-5 reranked results are passed to the LLM.

**Model size:** ~80MB · **Inference time:** ~150ms on CPU

---

### 3. Self-Corrective RAG

Before calling the LLM, the system scores its own retrieval quality across three signals:

| Signal | Method |
|---|---|
| **Retrieval score** | Average cosine similarity of top-k chunks |
| **Coverage score** | Fraction of query terms found in retrieved text |
| **Groundedness score** | Keyword overlap between answer and context |

**Decision logic:**
- `≥ 0.55` → proceed normally
- `0.30–0.55` → re-retrieve with expanded/reworded query
- `< 0.30` → re-retrieve + web search fallback (DuckDuckGo)

This reduces hallucinations by catching bad retrievals before they reach the LLM — without needing a second LLM call for grading.

---

### 4. Multi-Provider LLM with Circuit Breakers

The `MultiProviderLLMClient` manages three LLM providers with automatic failover:

```
Priority 1 → Groq (llama-3.3-70b-versatile)    fastest, free tier
Priority 2 → Google Gemini (gemini-2.0-flash)   1000 req/day free
Priority 3 → OpenRouter (free models)           unlimited fallback
```

**Circuit breaker logic:**
- Opens after **3 consecutive failures**
- Auto-recovers after **60 seconds**
- Each provider tracked independently

**Four routing strategies:** `priority` · `cost_aware` · `load_balance` · `fastest`

---

### 5. Agentic Mode (ReAct Loop)

A pure-Python ReAct (Reason-Act-Observe) loop without LangChain:

```
Thought → Action → Observation → Thought → ... → Final Answer
```

**Available tools:**
- `search_kb` — vector search over uploaded documents
- `web_search` — live DuckDuckGo search with multi-strategy fallback
- `summarize_doc` — summarise a specific indexed document

Runs up to 5 iterations, combining document knowledge with live web results into a single grounded answer.

---

### 6. Advanced Chunking Strategies

| Strategy | Algorithm | Best For |
|---|---|---|
| `semantic` | Splits at paragraph/heading boundaries | Most documents |
| `overlap` | Configurable % overlap (default 15%) | Dense technical docs |
| `dynamic` | Auto-detects document type → selects strategy | Mixed corpus |
| `sentence_window` | Embeds sentences, retrieves surrounding context | High-precision Q&A |

---

### 7. Query Transformation

| Technique | What it does |
|---|---|
| `multi_query` | Generates 3–5 alternative phrasings, searches all, deduplicates |
| `HyDE` | Generates a hypothetical ideal answer, embeds it for retrieval |
| `expansion` | Adds related terms to improve recall on sparse queries |

---

### 8. Multi-Tenant Isolation

| Level | Implementation | Compliance |
|---|---|---|
| `HIGH` | Separate ChromaDB directory per tenant | HIPAA, SOC2, FedRAMP |
| `MEDIUM` | Separate collection per tenant (same DB) | GDPR, SOC2 (default) |
| `LOW` | Shared collection + metadata filter | Basic separation |

Supports GDPR right-to-erasure: `manager.delete_tenant("tenant_id")` removes all data.

---

### 9. Automated Evaluation

50 built-in test queries across 5 categories (factual, reasoning, summary, comparison, multi-hop).

**Metrics scored per query:**
- **Retrieval Precision** — fraction of retrieved chunks that are relevant
- **Answer Correctness** — F1 keyword overlap with gold-standard answer
- **Groundedness** — how much of the answer is supported by retrieved context
- **Latency** — end-to-end and per-component timing

Run from the UI (Evaluation tab) or via code:

```python
from src.evaluation.eval_pipeline import EvaluationPipeline
evaluator = EvaluationPipeline(pipeline)
report = evaluator.run(queries, pass_threshold=0.6)
```

---

### 10. Structured Feedback & Priority Framework

When users click 👎, they're shown a failure category selector:

**Failure types:** `hallucination` · `retrieval_gap` · `partial_answer` · `citation_missing` · `off_topic` · `wrong_format` · `other`

The **Priority Framework** scores failures by:

```
Priority Score = Frequency × Persona Weight × Sensitivity Weight × Recency Decay
```

This produces a ranked action plan (Fix X before Y before Z) based on what actually matters to your users.

---

## 🖥️ User Interface

The Streamlit UI has 4 tabs:

### 💬 Chat Tab
- Real-time token streaming
- Per-message 👍/👎 feedback with failure categories
- Source cards with relevance score bars
- Confidence badge (self-correction mode)
- Agent reasoning steps (expandable)

### 📊 Analytics Tab
- Query latency (avg + P95)
- Token usage stats
- Failure type Pareto chart
- Priority action plan (ranked list of what to fix)
- Recent low-rated queries

### 🧪 Evaluation Tab
- Run built-in test suite (configurable: category, priority, count)
- Per-query score table (retrieval · correctness · groundedness · pass/fail)

### 🔌 Routing & Health Tab
- Live circuit breaker status for each LLM provider
- Manual circuit reset controls
- Strategy tester (send a query via any routing strategy)
- Tenant registration and isolation level management

---

## 🔑 API Key Modes

The sidebar supports two operating modes:

| Mode | Who uses it | How it works |
|---|---|---|
| **App's built-in key** | Developer / owner | Keys loaded from `.env` automatically |
| **Enter my own key** | External / public users | User pastes key into the UI — stored in browser session only, never written to disk |

Users can choose from Groq, Gemini, or OpenRouter — with placeholder text linking to where each free key can be obtained.

---

## 📡 REST API

```bash
uvicorn src.api.fastapi_app:app --port 8000
```

Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send a query, get answer + sources |
| `POST` | `/upload` | Upload and index a document |
| `GET` | `/health` | System health check |
| `GET` | `/metrics` | Prometheus-compatible metrics |
| `GET` | `/docs` | Interactive Swagger UI |

Rate limiting via SlowAPI. Authentication via JWT Bearer token.

---

## 📊 Observability Stack

```bash
docker-compose up prometheus grafana
```

| Service | URL | Purpose |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Main chatbot interface |
| FastAPI | http://localhost:8000/docs | REST API |
| Prometheus | http://localhost:9090 | Raw metrics |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Nginx | http://localhost:80 | Reverse proxy |

**Metrics tracked:**
- Query latency histogram (P50/P95/P99)
- Token usage counter per provider
- Retrieval quality gauge (avg similarity score)
- Error rate per provider
- Active user count
- Feedback satisfaction rate

---

## 🧪 Running Tests

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/test_chroma_store.py -v
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **UI** | Streamlit 1.35 |
| **REST API** | FastAPI + Uvicorn + SlowAPI |
| **LLM Providers** | Groq (Llama 3.3 70B) · Google Gemini 2.0 Flash · OpenRouter |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| **Vector DB** | ChromaDB 0.5 |
| **Sparse Search** | BM25Okapi (rank-bm25) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Web Search** | DuckDuckGo Search API |
| **Auth** | JWT (python-jose) + bcrypt (passlib) |
| **ORM / DB** | SQLAlchemy 2.0 + SQLite |
| **Metrics** | Prometheus Client + Grafana |
| **Logging** | Loguru |
| **Config** | Pydantic-Settings |
| **Document Parsing** | PyPDF2 · pdfplumber · python-docx · unstructured |
| **Testing** | pytest + pytest-cov |
| **Containerisation** | Docker + Docker Compose + Nginx |
| **Code Quality** | Black · isort · flake8 |

---

## 🔧 Configuration Reference

All settings can be set via `.env` or environment variables:

```env
# LLM
GROQ_API_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Vector DB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=rag_documents

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.1
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Upload
UPLOAD_DIR=./data/uploads
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=["pdf","txt","docx","md"]

# Auth
AUTH_DB_URL=sqlite:///./data/users.db
JWT_EXPIRE_MINUTES=1440

# App
DEBUG=false
LOG_LEVEL=INFO
```

---

## 📋 Requirements

- Python 3.10+
- ~2GB disk space (models + ChromaDB)
- ~1.5GB RAM (sentence-transformers + cross-encoder loaded in memory)
- No GPU required — all models run on CPU

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `pytest tests/ -v`
4. Format code: `black src/ && isort src/`
5. Open a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — ultra-fast LLM inference
- [ChromaDB](https://chromadb.com) — open-source vector database
- [Sentence-Transformers](https://sbert.net) — embedding and reranking models
- [Streamlit](https://streamlit.io) — rapid UI development
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation