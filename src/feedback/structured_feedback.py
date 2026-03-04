
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import sessionmaker

try:
    from sqlalchemy.orm import DeclarativeBase
    class Base(DeclarativeBase):
        pass
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()

from src.utils.logger import logger




class FailureType(str, Enum):
    HALLUCINATION     = "hallucination"
    CITATION_MISSING  = "citation_missing"
    PARTIAL_ANSWER    = "partial_answer"
    RETRIEVAL_GAP     = "retrieval_gap"
    OFF_TOPIC         = "off_topic"
    WRONG_FORMAT      = "wrong_format"
    OUTDATED_INFO     = "outdated_info"
    OTHER             = "other"


FAILURE_TYPE_LABELS = {
    FailureType.HALLUCINATION:    "🔴 Hallucination",
    FailureType.CITATION_MISSING: "🟡 Missing Citations",
    FailureType.PARTIAL_ANSWER:   "🟠 Partial Answer",
    FailureType.RETRIEVAL_GAP:    "🔵 Retrieval Gap",
    FailureType.OFF_TOPIC:        "⚪ Off-Topic",
    FailureType.WRONG_FORMAT:     "🟣 Wrong Format",
    FailureType.OUTDATED_INFO:    "⚫ Outdated Info",
    FailureType.OTHER:            "❔ Other",
}

FAILURE_TYPE_FIXES = {
    FailureType.HALLUCINATION:    "Improve groundedness scoring; lower similarity threshold; add fact-checking pass",
    FailureType.CITATION_MISSING: "Update system prompt to mandate citations; check retrieval is returning sources",
    FailureType.PARTIAL_ANSWER:   "Increase top_k results; improve context window size; check chunking strategy",
    FailureType.RETRIEVAL_GAP:    "Review chunking size; try different embedding model; add query expansion",
    FailureType.OFF_TOPIC:        "Improve system prompt; add query classification; check similarity threshold",
    FailureType.WRONG_FORMAT:     "Add format instructions to prompt; use structured output schemas",
    FailureType.OUTDATED_INFO:    "Update document corpus; add web search fallback for time-sensitive queries",
    FailureType.OTHER:            "Manual review required",
}




class StructuredFeedbackRecord(Base):
    __tablename__ = "structured_feedback"

    id                   = Column(String(36),   primary_key=True)
    user_id              = Column(String(36),   nullable=False, index=True)
    conversation_id      = Column(String(36),   nullable=True)
    query                = Column(Text,          nullable=False)
    answer_preview       = Column(Text,          nullable=True)
    rating               = Column(Integer,       nullable=False)
    failure_type         = Column(String(50),    nullable=True)
    expected_response    = Column(Text,          nullable=True)
    retrieved_doc_ids    = Column(JSON,          nullable=True)
    retrieved_doc_names  = Column(JSON,          nullable=True)
    retrieval_scores     = Column(JSON,          nullable=True)
    confidence_scores    = Column(JSON,          nullable=True)
    pipeline_metadata    = Column(JSON,          nullable=True)
    comment              = Column(Text,          nullable=True)
    latency_ms           = Column(Float,         nullable=True)
    model                = Column(String(64),    nullable=True)
    timestamp            = Column(DateTime,      default=datetime.utcnow)




class StructuredFeedbackCollector:


    def __init__(self, db_url: str = "sqlite:///./data/feedback.db") -> None:
        self.engine       = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info("[StructuredFeedback] Initialized with enhanced failure tracking")

    def record(
        self,
        user_id:             str,
        query:               str,
        answer:              str,
        rating:              int,
        failure_type:        Optional[FailureType] = None,
        expected_response:   Optional[str]         = None,
        retrieved_doc_ids:   Optional[List[str]]   = None,
        retrieved_doc_names: Optional[List[str]]   = None,
        retrieval_scores:    Optional[List[float]]  = None,
        confidence_scores:   Optional[Dict[str, float]] = None,
        pipeline_metadata:   Optional[Dict]         = None,
        conversation_id:     Optional[str]          = None,
        comment:             Optional[str]          = None,
        latency_ms:          Optional[float]        = None,
        model:               Optional[str]          = None,
    ) -> str:

        db = self.SessionLocal()
        try:
            record = StructuredFeedbackRecord(
                id                   = str(uuid4()),
                user_id              = user_id,
                conversation_id      = conversation_id,
                query                = query,
                answer_preview       = answer[:500] if answer else "",
                rating               = rating,
                failure_type         = failure_type.value if failure_type else None,
                expected_response    = expected_response,
                retrieved_doc_ids    = retrieved_doc_ids or [],
                retrieved_doc_names  = retrieved_doc_names or [],
                retrieval_scores     = retrieval_scores or [],
                confidence_scores    = confidence_scores or {},
                pipeline_metadata    = pipeline_metadata or {},
                comment              = comment,
                latency_ms           = latency_ms,
                model                = model,
            )
            db.add(record)
            db.commit()

            label = FAILURE_TYPE_LABELS.get(failure_type, "") if failure_type else ""
            logger.info(
                f"[StructuredFeedback] Recorded: user={user_id} "
                f"rating={'👍' if rating > 0 else '👎'} "
                f"failure={label} "
                f"query='{query[:40]}'"
            )
            return record.id
        finally:
            db.close()

    def get_failure_analysis(
        self,
        user_id:    Optional[str] = None,
        last_n:     int            = 200,
    ) -> Dict[str, Any]:

        db = self.SessionLocal()
        try:
            q = db.query(StructuredFeedbackRecord)
            if user_id:
                q = q.filter(StructuredFeedbackRecord.user_id == user_id)

            records = q.order_by(
                StructuredFeedbackRecord.timestamp.desc()
            ).limit(last_n).all()

            if not records:
                return {"total": 0, "failures": {}, "top_issues": [], "recommended_fixes": []}

            total     = len(records)
            negative  = [r for r in records if r.rating < 0]
            positive  = total - len(negative)


            failure_counts: Dict[str, int] = {}
            for r in negative:
                ft = r.failure_type or FailureType.OTHER.value
                failure_counts[ft] = failure_counts.get(ft, 0) + 1


            sorted_failures = sorted(
                failure_counts.items(), key=lambda x: x[1], reverse=True
            )


            recommendations = []
            for ft_val, count in sorted_failures[:5]:
                try:
                    ft = FailureType(ft_val)
                    recommendations.append({
                        "failure_type": ft_val,
                        "label":        FAILURE_TYPE_LABELS.get(ft, ft_val),
                        "count":        count,
                        "percentage":   round(count / max(len(negative), 1) * 100, 1),
                        "fix":          FAILURE_TYPE_FIXES.get(ft, "Review manually"),
                    })
                except ValueError:
                    pass


            doc_failures: Dict[str, int] = {}
            for r in negative:
                if r.retrieved_doc_names:
                    for doc in r.retrieved_doc_names:
                        doc_failures[doc] = doc_failures.get(doc, 0) + 1


            avg_confidence: Dict[str, float] = {}
            confidence_records = [r for r in negative if r.confidence_scores]
            if confidence_records:
                all_keys = set()
                for r in confidence_records:
                    all_keys.update(r.confidence_scores.keys())
                for key in all_keys:
                    vals = [r.confidence_scores[key] for r in confidence_records
                            if key in r.confidence_scores]
                    avg_confidence[key] = round(sum(vals) / len(vals), 3)

            return {
                "total":              total,
                "positive":           positive,
                "negative":           len(negative),
                "satisfaction_rate":  round(positive / total * 100, 1),
                "failure_breakdown":  failure_counts,
                "top_issues":         sorted_failures[:5],
                "recommended_fixes":  recommendations,
                "worst_documents":    sorted(doc_failures.items(), key=lambda x: -x[1])[:5],
                "avg_failure_confidence": avg_confidence,
                "has_corrections":    sum(1 for r in negative if r.expected_response),
            }
        finally:
            db.close()

    def get_recent_failures(
        self,
        user_id:      Optional[str]      = None,
        failure_type: Optional[FailureType] = None,
        limit:        int                  = 20,
    ) -> List[Dict]:
        """Get recent failed queries with full context for debugging."""
        db = self.SessionLocal()
        try:
            q = db.query(StructuredFeedbackRecord).filter(
                StructuredFeedbackRecord.rating < 0
            )
            if user_id:
                q = q.filter(StructuredFeedbackRecord.user_id == user_id)
            if failure_type:
                q = q.filter(StructuredFeedbackRecord.failure_type == failure_type.value)

            records = q.order_by(
                StructuredFeedbackRecord.timestamp.desc()
            ).limit(limit).all()

            return [
                {
                    "id":                r.id,
                    "query":             r.query[:100],
                    "answer_preview":    r.answer_preview[:200] if r.answer_preview else "",
                    "failure_type":      r.failure_type,
                    "failure_label":     FAILURE_TYPE_LABELS.get(
                        FailureType(r.failure_type) if r.failure_type else FailureType.OTHER, ""
                    ),
                    "expected_response": r.expected_response,
                    "retrieved_docs":    r.retrieved_doc_names or [],
                    "retrieval_scores":  r.retrieval_scores or [],
                    "confidence_scores": r.confidence_scores or {},
                    "comment":           r.comment,
                    "timestamp":         str(r.timestamp),
                    "suggested_fix":     FAILURE_TYPE_FIXES.get(
                        FailureType(r.failure_type) if r.failure_type else FailureType.OTHER, ""
                    ),
                }
                for r in records
            ]
        finally:
            db.close()



_collector: Optional[StructuredFeedbackCollector] = None

def get_structured_feedback() -> StructuredFeedbackCollector:
    global _collector
    if _collector is None:
        _collector = StructuredFeedbackCollector()
    return _collector
