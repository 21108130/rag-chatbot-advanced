
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field




class DocumentStatus(str, Enum):
    PENDING   = "pending"
    PROCESSING = "processing"
    INDEXED   = "indexed"
    FAILED    = "failed"


class MessageRole(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"




class DocumentChunk(BaseModel):

    chunk_id:    str = Field(default_factory=lambda: str(uuid4()))
    doc_id:      str
    content:     str
    chunk_index: int
    metadata:    Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class Document(BaseModel):

    doc_id:      str = Field(default_factory=lambda: str(uuid4()))
    filename:    str
    file_type:   str
    file_size:   int                          # bytes
    status:      DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    created_at:  datetime = Field(default_factory=datetime.utcnow)
    metadata:    Dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResult(BaseModel):

    doc_id:      str
    filename:    str
    status:      DocumentStatus
    chunk_count: int
    message:     str


class RetrievedChunk(BaseModel):

    chunk_id:         str
    doc_id:           str
    content:          str
    similarity_score: float
    metadata:         Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):

    query:   str
    chunks:  List[RetrievedChunk]
    latency_ms: float = 0.0


class ChatMessage(BaseModel):

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    role:       MessageRole
    content:    str
    timestamp:  datetime = Field(default_factory=datetime.utcnow)
    metadata:   Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query:           str
    conversation_id: Optional[str] = None
    top_k:           Optional[int] = None


class ChatResponse(BaseModel):

    answer:          str
    sources:         List[RetrievedChunk]
    conversation_id: str
    latency_ms:      float
    tokens_used:     Optional[int] = None


class ConversationHistory(BaseModel):

    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    messages:        List[ChatMessage] = Field(default_factory=list)

    def add_message(self, role: MessageRole, content: str) -> ChatMessage:
        msg = ChatMessage(role=role, content=content)
        self.messages.append(msg)
        return msg

    def get_context_window(self, last_n: int = 10) -> List[ChatMessage]:

        return self.messages[-last_n:]

    def to_llm_messages(self, last_n: int = 10) -> List[Dict[str, str]]:

        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.get_context_window(last_n)
        ]
