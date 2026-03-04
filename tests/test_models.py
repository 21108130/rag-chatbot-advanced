
import pytest
from src.utils.models import (
    ChatMessage,
    ConversationHistory,
    DocumentChunk,
    DocumentStatus,
    MessageRole,
    RetrievedChunk,
)


class TestDocumentChunk:

    def test_auto_id_generated(self):
        chunk = DocumentChunk(doc_id="d1", content="hello", chunk_index=0)
        assert chunk.chunk_id
        assert len(chunk.chunk_id) == 36  

    def test_immutable(self):
        chunk = DocumentChunk(doc_id="d1", content="hello", chunk_index=0)
        with pytest.raises(Exception):
            chunk.content = "changed"


class TestConversationHistory:

    def setup_method(self):
        self.history = ConversationHistory()

    def test_add_message(self):
        msg = self.history.add_message(MessageRole.USER, "Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert len(self.history.messages) == 1

    def test_context_window_truncated(self):
        for i in range(20):
            self.history.add_message(MessageRole.USER, f"msg {i}")
        window = self.history.get_context_window(last_n=5)
        assert len(window) == 5

    def test_to_llm_messages_format(self):
        self.history.add_message(MessageRole.USER,      "question")
        self.history.add_message(MessageRole.ASSISTANT, "answer")
        msgs = self.history.to_llm_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert all("content" in m for m in msgs)


class TestDocumentStatus:

    def test_enum_values(self):
        assert DocumentStatus.INDEXED  == "indexed"
        assert DocumentStatus.FAILED   == "failed"
        assert DocumentStatus.PENDING  == "pending"
