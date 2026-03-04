
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.embedder import EmbeddingGenerator


@pytest.fixture
def mock_model():

    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    model.encode.return_value = np.random.rand(384).astype(np.float32)
    return model


@pytest.fixture
def embedder(mock_model):
    gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    gen._model = mock_model
    return gen


class TestEmbeddingGenerator:

    def test_embed_text_returns_vector(self, embedder):
        vec = embedder.embed_text("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)

    def test_empty_text_returns_zero_vector(self, embedder):
        vec = embedder.embed_text("")
        assert np.all(vec == 0)

    def test_caching_avoids_re_encoding(self, embedder):
        text = "Cache me"
        embedder.embed_text(text)
        embedder.embed_text(text)
      
        assert embedder._model.encode.call_count == 1

    def test_embed_batch_returns_list(self, embedder, mock_model):
        texts = ["text 1", "text 2", "text 3"]
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        vecs = embedder.embed_batch(texts, show_progress=False)
        assert len(vecs) == 3
        assert all(v.shape == (384,) for v in vecs)

    def test_clear_cache(self, embedder):
        embedder.embed_text("clear me")
        assert len(embedder._cache) > 0
        embedder.clear_cache()
        assert len(embedder._cache) == 0

    def test_dimension_property(self, embedder):
        assert embedder.dimension == 384
