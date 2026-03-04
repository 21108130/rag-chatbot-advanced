
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import json

from src.evaluation.eval_pipeline import EvalQuery




SAMPLE_EVAL_QUERIES: List[EvalQuery] = [


    EvalQuery("F001", "What is Retrieval Augmented Generation?",
              "RAG combines retrieval of relevant documents with language model generation to produce grounded answers",
              ["retrieval", "augmented", "generation", "RAG", "documents"],
              "factual", "high"),

    EvalQuery("F002", "What is an agentic AI system?",
              "Agentic AI systems can plan, reason, use tools, and take autonomous actions to complete complex tasks",
              ["agent", "autonomous", "planning", "tools", "actions"],
              "factual", "high"),

    EvalQuery("F003", "What is the difference between RAG and fine-tuning?",
              "RAG retrieves external knowledge at inference time while fine-tuning updates model weights during training",
              ["fine-tuning", "retrieval", "knowledge", "training", "inference"],
              "factual", "high"),

    EvalQuery("F004", "What is a vector database?",
              "A vector database stores embeddings and supports similarity search to find semantically similar content",
              ["vector", "embeddings", "similarity", "search", "database"],
              "factual", "high"),

    EvalQuery("F005", "What are embeddings in NLP?",
              "Embeddings are dense numerical vector representations of text that capture semantic meaning",
              ["embeddings", "vector", "semantic", "representation"],
              "factual", "high"),

    EvalQuery("F006", "What is semantic search?",
              "Semantic search finds content based on meaning and context rather than exact keyword matching",
              ["semantic", "meaning", "context", "search"],
              "factual", "medium"),

    EvalQuery("F007", "What is a language model?",
              "A language model is trained to predict and generate text, learning patterns from large text corpora",
              ["language model", "text", "generation", "training"],
              "factual", "medium"),

    EvalQuery("F008", "What is ChromaDB?",
              "ChromaDB is an open-source vector database for storing and querying embeddings with persistence",
              ["ChromaDB", "vector", "embeddings", "database"],
              "factual", "medium"),

    EvalQuery("F009", "What is the ReAct framework for AI agents?",
              "ReAct combines reasoning and acting, alternating between thinking steps and tool use actions",
              ["ReAct", "reasoning", "acting", "tools", "agent"],
              "factual", "high"),

    EvalQuery("F010", "What are transformer models?",
              "Transformers use self-attention mechanisms to process sequences and are the basis for modern LLMs",
              ["transformer", "attention", "neural network", "sequence"],
              "factual", "medium"),


    EvalQuery("R001", "Why is chunking important in RAG systems?",
              "Chunking splits documents into smaller pieces so they fit in context windows and improve retrieval precision",
              ["chunk", "retrieval", "context", "documents"],
              "reasoning", "high"),

    EvalQuery("R002", "How does hybrid search improve retrieval quality?",
              "Hybrid search combines dense vector search with sparse keyword BM25 search to capture both semantic and lexical matches",
              ["hybrid", "vector", "BM25", "retrieval", "keyword"],
              "reasoning", "high"),

    EvalQuery("R003", "Why is reranking beneficial after initial retrieval?",
              "Reranking uses a more powerful cross-encoder model to re-score chunks, improving relevance ordering",
              ["reranking", "cross-encoder", "relevance", "retrieval"],
              "reasoning", "high"),

    EvalQuery("R004", "What are the limitations of basic RAG?",
              "Basic RAG may retrieve irrelevant chunks, lose context at chunk boundaries, and cannot handle multi-hop reasoning",
              ["limitation", "retrieval", "context", "hallucination"],
              "reasoning", "high"),

    EvalQuery("R005", "How do agents differ from standard LLM chatbots?",
              "Agents can use tools, search external sources, take multiple steps, and make decisions autonomously",
              ["agent", "tools", "autonomous", "multi-step", "decision"],
              "reasoning", "high"),

    EvalQuery("R006", "What causes hallucinations in LLMs?",
              "LLMs hallucinate when they generate plausible-sounding but unsupported text not grounded in retrieved context",
              ["hallucination", "grounded", "context", "unsupported"],
              "reasoning", "high"),

    EvalQuery("R007", "Why is context window size important for RAG?",
              "The context window limits how much retrieved content can be passed to the LLM for answer generation",
              ["context window", "tokens", "LLM", "retrieval"],
              "reasoning", "medium"),

    EvalQuery("R008", "How does query expansion help retrieval?",
              "Query expansion adds related terms and alternative phrasings to improve recall of relevant documents",
              ["query expansion", "recall", "terms", "retrieval"],
              "reasoning", "medium"),

    EvalQuery("R009", "What is the role of embeddings in semantic similarity?",
              "Embeddings map text to vector space where semantically similar texts have high cosine similarity",
              ["embeddings", "cosine similarity", "semantic", "vector"],
              "reasoning", "medium"),

    EvalQuery("R010", "Why use multi-query generation for retrieval?",
              "Multiple query phrasings retrieve diverse chunks, reducing the chance of missing relevant information",
              ["multi-query", "phrasings", "diverse", "retrieval"],
              "reasoning", "medium"),

    EvalQuery("S001", "Summarize the key components of a RAG pipeline",
              "RAG pipeline includes document loading, chunking, embedding, vector storage, retrieval, reranking, and LLM generation",
              ["pipeline", "retrieval", "embedding", "generation", "components"],
              "summary", "medium"),

    EvalQuery("S002", "What are the main challenges in building production RAG systems?",
              "Challenges include chunking strategy, retrieval quality, hallucination prevention, latency, cost, and multi-tenancy",
              ["production", "challenges", "quality", "latency", "cost"],
              "summary", "medium"),

    EvalQuery("S003", "Describe the agentic AI workflow",
              "Agentic workflow involves planning, tool selection, execution, observation, and iterative refinement",
              ["agentic", "planning", "tools", "execution", "workflow"],
              "summary", "medium"),

    EvalQuery("S004", "What evaluation metrics matter for RAG systems?",
              "Key metrics include retrieval precision, answer correctness, groundedness, hallucination rate, and latency",
              ["evaluation", "metrics", "precision", "correctness", "latency"],
              "summary", "high"),

    EvalQuery("S005", "How does the reranking stage work in an advanced RAG pipeline?",
              "After initial retrieval, a cross-encoder model scores each query-chunk pair for precise relevance ordering",
              ["reranking", "cross-encoder", "score", "relevance", "query"],
              "summary", "medium"),

    EvalQuery("C001", "Compare BM25 and vector search for document retrieval",
              "BM25 is keyword-based and exact while vector search is semantic; hybrid combines both strengths",
              ["BM25", "vector", "keyword", "semantic", "hybrid"],
              "comparison", "high"),

    EvalQuery("C002", "What is the difference between a bi-encoder and cross-encoder?",
              "Bi-encoders embed query and document separately while cross-encoders jointly encode them for higher accuracy",
              ["bi-encoder", "cross-encoder", "embedding", "accuracy"],
              "comparison", "high"),

    EvalQuery("C003", "How do RAG and long-context models compare?",
              "RAG retrieves relevant chunks while long-context models process entire documents but at higher cost",
              ["RAG", "long-context", "retrieval", "cost", "context"],
              "comparison", "medium"),

    EvalQuery("C004", "What are the tradeoffs between chunk size and retrieval quality?",
              "Smaller chunks are more precise but lose context, larger chunks preserve context but reduce precision",
              ["chunk size", "precision", "context", "tradeoff"],
              "comparison", "medium"),

    EvalQuery("C005", "Compare streaming vs standard LLM generation",
              "Streaming yields tokens progressively improving perceived latency while standard waits for complete response",
              ["streaming", "latency", "tokens", "generation"],
              "comparison", "low"),


    EvalQuery("M001", "How does query transformation improve retrieval, and what techniques exist?",
              "Query transformation includes multi-query generation, HyDE, and expansion; each improves recall differently",
              ["query transformation", "HyDE", "multi-query", "expansion", "recall"],
              "multi-hop", "high"),

    EvalQuery("M002", "How do evaluation results drive system improvement in a RAG pipeline?",
              "Evaluation identifies weak retrieval, poor chunking, and hallucinations, enabling targeted configuration changes",
              ["evaluation", "improvement", "retrieval", "chunking", "configuration"],
              "multi-hop", "high"),

    EvalQuery("M003", "What role does user feedback play in RAG system optimization?",
              "User feedback identifies failure categories and high-impact queries for prioritized system improvements",
              ["feedback", "optimization", "failure", "improvement", "priority"],
              "multi-hop", "high"),

    EvalQuery("M004", "How does self-corrective RAG reduce hallucinations?",
              "Self-corrective RAG scores retrieved documents and re-retrieves when confidence is low before generation",
              ["self-corrective", "confidence", "re-retrieval", "hallucination"],
              "multi-hop", "high"),

    EvalQuery("M005", "Explain how multi-tenant isolation works in a production RAG system",
              "Multi-tenancy uses separate vector collections per user or tenant to prevent cross-user data leakage",
              ["multi-tenant", "isolation", "collection", "user", "security"],
              "multi-hop", "high"),

    EvalQuery("F011", "What is cosine similarity?",
              "Cosine similarity measures the angle between two vectors, returning 1 for identical direction and 0 for orthogonal",
              ["cosine", "similarity", "vector", "angle"],
              "factual", "low"),

    EvalQuery("F012", "What is a prompt template?",
              "A prompt template is a structured text format with placeholders filled at runtime to guide LLM responses",
              ["prompt", "template", "LLM", "structured"],
              "factual", "low"),

    EvalQuery("F013", "What is LangChain?",
              "LangChain is a framework for building LLM applications with chains, agents, and retrieval components",
              ["LangChain", "framework", "LLM", "chains", "agents"],
              "factual", "low"),

    EvalQuery("F014", "What is Groq?",
              "Groq provides fast LLM inference API using custom hardware for low-latency language model responses",
              ["Groq", "inference", "API", "fast", "LLM"],
              "factual", "medium"),

    EvalQuery("F015", "What is SentenceTransformers?",
              "SentenceTransformers provides pre-trained models for generating sentence and document embeddings",
              ["SentenceTransformers", "embeddings", "models", "sentence"],
              "factual", "medium"),

    EvalQuery("R011", "How does context optimization reduce LLM costs?",
              "Context optimization removes duplicate chunks and truncates to token budget, reducing tokens sent to LLM",
              ["context optimization", "tokens", "cost", "deduplication", "truncation"],
              "reasoning", "high"),

    EvalQuery("R012", "What is reciprocal rank fusion?",
              "RRF combines rankings from multiple retrieval methods by summing reciprocal rank scores for each document",
              ["RRF", "rank fusion", "retrieval", "ranking", "score"],
              "reasoning", "high"),

    EvalQuery("R013", "Why is document extraction quality critical for RAG?",
              "Poor extraction introduces noise and boilerplate that degrades embedding quality and retrieval precision",
              ["extraction", "quality", "noise", "embeddings", "retrieval"],
              "reasoning", "high"),

    EvalQuery("R014", "How does HyDE improve retrieval accuracy?",
              "HyDE generates a hypothetical ideal answer first, then uses it as the query vector for more relevant retrieval",
              ["HyDE", "hypothetical", "answer", "retrieval", "accuracy"],
              "reasoning", "high"),

    EvalQuery("S006", "What is the purpose of a circuit breaker in multi-provider LLM routing?",
              "A circuit breaker temporarily disables a failing provider to prevent cascading failures and improve reliability",
              ["circuit breaker", "provider", "failure", "reliability", "routing"],
              "summary", "high"),

    EvalQuery("C006", "What is the difference between precision and recall in retrieval?",
              "Precision is the fraction of retrieved documents that are relevant; recall is the fraction of relevant documents retrieved",
              ["precision", "recall", "retrieved", "relevant", "fraction"],
              "comparison", "high"),

]


def load_eval_dataset(
    category: Optional[str] = None,
    priority: Optional[str] = None,
    max_queries: Optional[int] = None,
) -> List[EvalQuery]:

    queries = SAMPLE_EVAL_QUERIES

    if category:
        queries = [q for q in queries if q.category == category]
    if priority:
        queries = [q for q in queries if q.priority == priority]
    if max_queries:
        queries = queries[:max_queries]

    return queries


def load_eval_dataset_from_json(path: str) -> List[EvalQuery]:

    with open(path) as f:
        data = json.load(f)
    return [
        EvalQuery(
            query_id              = q["query_id"],
            query                 = q["query"],
            expected_answer       = q["expected_answer"],
            relevant_doc_keywords = q.get("relevant_doc_keywords", []),
            category              = q.get("category", "general"),
            priority              = q.get("priority", "medium"),
        )
        for q in data
    ]
