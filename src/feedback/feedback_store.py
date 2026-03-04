
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import sessionmaker

try:
    from sqlalchemy.orm import DeclarativeBase
    class Base(DeclarativeBase):
        pass
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()

from src.utils.logger import logger


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id                    = Column(String(36),  primary_key=True, default=lambda: str(uuid4()))
    user_id               = Column(String(36),  nullable=False, index=True)
    conversation_id       = Column(String(36),  nullable=True)
    query                 = Column(Text,         nullable=False)
    answer_preview        = Column(Text,         nullable=True)
    rating                = Column(Integer,      nullable=False)
    comment               = Column(Text,         nullable=True)
    retrieved_chunks_count = Column(Integer,     nullable=True)
    latency_ms            = Column(Float,        nullable=True)
    model                 = Column(String(64),   nullable=True)
    timestamp             = Column(DateTime,     default=datetime.utcnow)


class FeedbackStore:


    def __init__(self, db_url: str = "sqlite:///./data/feedback.db") -> None:
        self.engine       = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info("[Feedback] Feedback store initialized.")

    def record(
        self,
        user_id:                str,
        query:                  str,
        rating:                 int,   # 1 or -1
        answer:                 str   = "",
        conversation_id:        Optional[str]  = None,
        comment:                Optional[str]  = None,
        retrieved_chunks_count: Optional[int]  = None,
        latency_ms:             Optional[float] = None,
        model:                  Optional[str]  = None,
    ) -> str:

        db = self.SessionLocal()
        try:
            record = FeedbackRecord(
                id                     = str(uuid4()),
                user_id                = user_id,
                conversation_id        = conversation_id,
                query                  = query,
                answer_preview         = answer[:500] if answer else "",
                rating                 = rating,
                comment                = comment,
                retrieved_chunks_count = retrieved_chunks_count,
                latency_ms             = latency_ms,
                model                  = model,
            )
            db.add(record)
            db.commit()
            logger.info(
                f"[Feedback] Recorded: user={user_id} "
                f"rating={'👍' if rating > 0 else '👎'} "
                f"query='{query[:40]}'"
            )
            return record.id
        finally:
            db.close()

    def get_summary(
        self,
        user_id:  Optional[str] = None,
        last_n:   int            = 100,
    ) -> Dict[str, Any]:

        db = self.SessionLocal()
        try:
            query = db.query(FeedbackRecord)
            if user_id:
                query = query.filter(FeedbackRecord.user_id == user_id)
            records = query.order_by(FeedbackRecord.timestamp.desc()).limit(last_n).all()

            if not records:
                return {"total": 0, "positive": 0, "negative": 0, "satisfaction_rate": 0.0}

            total    = len(records)
            positive = sum(1 for r in records if r.rating > 0)
            negative = total - positive
            avg_latency = (
                sum(r.latency_ms for r in records if r.latency_ms) /
                max(1, sum(1 for r in records if r.latency_ms))
            )


            bad_queries = [
                {"query": r.query[:80], "comment": r.comment, "ts": str(r.timestamp)}
                for r in records if r.rating < 0
            ][:10]

            return {
                "total":             total,
                "positive":          positive,
                "negative":          negative,
                "satisfaction_rate": round(positive / total * 100, 1),
                "avg_latency_ms":    round(avg_latency, 1),
                "bad_queries":       bad_queries,
            }
        finally:
            db.close()

    def get_recent(
        self,
        user_id: Optional[str] = None,
        limit:   int            = 20,
    ) -> List[Dict]:
        """Return recent feedback records."""
        db = self.SessionLocal()
        try:
            query = db.query(FeedbackRecord)
            if user_id:
                query = query.filter(FeedbackRecord.user_id == user_id)
            records = query.order_by(FeedbackRecord.timestamp.desc()).limit(limit).all()
            return [
                {
                    "id":        r.id,
                    "query":     r.query[:80],
                    "rating":    "👍" if r.rating > 0 else "👎",
                    "comment":   r.comment,
                    "timestamp": str(r.timestamp),
                }
                for r in records
            ]
        finally:
            db.close()




_feedback_store: Optional[FeedbackStore] = None

def get_feedback_store() -> FeedbackStore:
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store
