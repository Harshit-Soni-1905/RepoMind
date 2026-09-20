"""Tests for embedding generator."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from repomind.vectorstore.embedder import Embedder
from repomind.config import config


def test_embedder_embed_single_text():
    """Embedder should generate embedding for a single text string."""
    embedder = Embedder()
    text = "def hello(): pass"

    embedding = embedder.embed(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1  # 1D array
    assert len(embedding) == config.EMBEDDING_DIMENSION


def test_embedder_embed_batch():
    """Embedder should generate embeddings for multiple texts."""
    embedder = Embedder()
    texts = [
        "def hello(): pass",
        "class Foo: pass",
        "import os",
    ]

    embeddings = embedder.embed_batch(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, config.EMBEDDING_DIMENSION)


def test_embedder_empty_batch():
    """Embedder should handle empty batch gracefully."""
    embedder = Embedder()
    embeddings = embedder.embed_batch([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0,)


def test_embedder_dimension_property():
    """Embedder should report correct embedding dimension."""
    embedder = Embedder()
    dimension = embedder.dimension

    assert dimension == config.EMBEDDING_DIMENSION
    assert dimension == 384  # all-MiniLM-L6-v2 produces 384-dim vectors


def test_embedder_different_texts_different_embeddings():
    """Different texts should produce different embeddings."""
    embedder = Embedder()
    text1 = "def hello(): pass"
    text2 = "class Foo: pass"

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)

    # Embeddings should not be identical
    assert not np.allclose(emb1, emb2)


def test_embedder_same_text_same_embedding():
    """Same text should produce same embedding."""
    embedder = Embedder()
    text = "def hello(): pass"

    emb1 = embedder.embed(text)
    emb2 = embedder.embed(text)

    # Should be identical (deterministic)
    assert np.allclose(emb1, emb2)


def test_embedder_model_caching():
    """Multiple Embedder instances should share the same loaded model."""
    embedder1 = Embedder()
    _ = embedder1.embed("test")

    embedder2 = Embedder()
    _ = embedder2.embed("test")

    # Both should reference the same cached model
    assert Embedder._model is not None
    assert embedder1._model is embedder2._model


def test_embedder_normalized_vectors():
    """Embeddings from all-MiniLM-L6-v2 should be normalized (unit vectors)."""
    embedder = Embedder()
    text = "def compute_sum(a, b): return a + b"

    embedding = embedder.embed(text)
    norm = np.linalg.norm(embedding)

    # all-MiniLM-L6-v2 produces normalized embeddings (L2 norm ≈ 1.0)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_embedder_custom_model_name():
    """Embedder should accept custom model name."""
    # Use the same model but verify the parameter is accepted
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text = "def foo(): pass"

    embedding = embedder.embed(text)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384


def test_embedder_semantic_similarity():
    """Semantically similar code should produce similar embeddings."""
    embedder = Embedder()

    # Similar texts
    text1 = "def calculate_sum(x, y): return x + y"
    text2 = "def add_numbers(a, b): return a + b"

    # Different text
    text3 = "class Database: pass"

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)
    emb3 = embedder.embed(text3)

    # Cosine similarity: dot product of normalized vectors
    sim_12 = np.dot(emb1, emb2)
    sim_13 = np.dot(emb1, emb3)

    # Similar texts should have higher similarity
    assert sim_12 > sim_13


# ---- Batching regression tests ----

def test_embedder_batch_smaller_than_batch_size():
    """embed_batch with fewer texts than batch_size should work correctly."""
    embedder = Embedder()
    texts = [
        "def hello(): pass",
        "class Foo: pass",
    ]  # 2 texts, batch_size default 32

    embeddings = embedder.embed_batch(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, config.EMBEDDING_DIMENSION)


def test_embedder_batch_larger_than_batch_size():
    """embed_batch with more texts than batch_size should process in multiple batches."""
    embedder = Embedder()
    # 100 texts, batch_size default 32 -> 4 batches (32, 32, 32, 4)
    texts = [f"def func_{i}(): pass" for i in range(100)]

    embeddings = embedder.embed_batch(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (100, config.EMBEDDING_DIMENSION)


def test_embedder_batch_correct_order_and_shape():
    """Embeddings should be in same order as input with correct shape."""
    embedder = Embedder()
    texts = [f"def func_{i}(): return {i}" for i in range(50)]

    embeddings = embedder.embed_batch(texts, batch_size=16)

    assert embeddings.shape == (50, config.EMBEDDING_DIMENSION)
    # Verify order preserved - each embedding should be different
    for i in range(1, len(embeddings)):
        assert not np.allclose(embeddings[i], embeddings[i-1])


def test_embedder_batch_empty_input():
    """embed_batch with empty list should return empty array."""
    embedder = Embedder()
    embeddings = embedder.embed_batch([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0,)


def test_embedder_batch_calls_model_multiple_times(monkeypatch):
    """embed_batch should call underlying model multiple times when input exceeds batch_size."""
    embedder = Embedder()
    # Force model to load
    _ = embedder.dimension

    call_count = 0

    def mock_encode(batch, convert_to_numpy=True):
        nonlocal call_count
        call_count += 1
        # Return fake embeddings of correct shape
        return np.random.rand(len(batch), config.EMBEDDING_DIMENSION).astype(np.float32)

    with patch.object(Embedder._model, 'encode', side_effect=mock_encode):
        texts = [f"text_{i}" for i in range(100)]
        embeddings = embedder.embed_batch(texts, batch_size=32)

    # 100 texts with batch_size=32 should call encode 4 times (32+32+32+4)
    assert call_count == 4
    assert embeddings.shape == (100, config.EMBEDDING_DIMENSION)


def test_embedder_batch_custom_batch_size():
    """embed_batch should respect custom batch_size parameter."""
    embedder = Embedder()
    texts = [f"def func_{i}(): pass" for i in range(10)]

    embeddings = embedder.embed_batch(texts, batch_size=3)

    assert embeddings.shape == (10, config.EMBEDDING_DIMENSION)
