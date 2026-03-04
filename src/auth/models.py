
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, String, Text
try:
    from sqlalchemy.orm import DeclarativeBase
    class Base(DeclarativeBase):
        pass
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id            = Column(String(36), primary_key=True)
    username      = Column(String(64), unique=True, nullable=False, index=True)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    last_login    = Column(DateTime, nullable=True)
  
    collection_name = Column(String(128), unique=True, nullable=False)

    def __repr__(self):
        return f"<User id={self.id} username={self.username}>"


class APIKey(Base):
    __tablename__ = "api_keys"

    id         = Column(String(36), primary_key=True)
    user_id    = Column(String(36), nullable=False, index=True)
    key_hash   = Column(String(255), unique=True, nullable=False)
    name       = Column(String(128), nullable=False)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used  = Column(DateTime, nullable=True)
