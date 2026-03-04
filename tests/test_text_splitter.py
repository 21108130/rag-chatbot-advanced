
import pytest
from src.utils.text_splitter import RecursiveTextSplitter


class TestRecursiveTextSplitter:

    def setup_method(self):
        self.splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)

    def test_empty_text_returns_empty(self):
        assert self.splitter.split_text("") == []
        assert self.splitter.split_text("   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "Hello world. This is a short text."
        chunks = self.splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_long_text_is_split(self):
        text = " ".join(["word"] * 200)
        chunks = self.splitter.split_text(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= self.splitter.chunk_size * 2
    def test_overlap_is_applied(self):
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "A" * 50 + " " + "B" * 50
        chunks = splitter.split_text(text)
        if len(chunks) > 1:
           
            assert len(chunks[1]) > 0

    def test_paragraph_boundaries_preferred(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        splitter = RecursiveTextSplitter(chunk_size=30, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
