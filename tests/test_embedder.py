"""Tests for embedding generator using FastEmbed (ONNX Runtime)."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from repomind.vectorstore.embedder import Embedder
from repomind.config import config


def test_embedder_embed_single_text_shape_and_dtype():
    """Embedder should generate a float32 1D embedding array of dimension 384."""
    embedder = Embedder()
    text = "def hello(): pass"

    embedding = embedder.embed(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (config.EMBEDDING_DIMENSION,)
    assert embedding.dtype == np.float32


def test_embedder_embed_batch_shape_and_dtype():
    """Embedder should generate float32 2D embeddings of shape (N, 384)."""
    embedder = Embedder()
    texts = [
        "def hello(): pass",
        "class Foo: pass",
        "import os",
    ]

    embeddings = embedder.embed_batch(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, config.EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32


def test_embedder_empty_batch_behavior():
    """Empty batch should return a 2D float32 array of shape (0, 384)."""
    embedder = Embedder()
    embeddings = embedder.embed_batch([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, config.EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32


def test_embedder_dimension_property_does_not_require_model_load():
    """Dimension property should return EMBEDDING_DIMENSION without model loading."""
    # Reset singleton model to test property behavior
    old_model = Embedder._model
    old_name = Embedder._model_name
    try:
        Embedder._model = None
        Embedder._model_name = None
        embedder = Embedder()
        dimension = embedder.dimension

        assert dimension == config.EMBEDDING_DIMENSION
        assert dimension == 384
        # Model should still not be loaded for default model
        assert Embedder._model is None
    finally:
        Embedder._model = old_model
        Embedder._model_name = old_name


def test_embedder_different_texts_produce_different_embeddings():
    """Different texts should produce distinct embeddings."""
    embedder = Embedder()
    text1 = "def hello(): pass"
    text2 = "class Foo: pass"

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)

    assert not np.allclose(emb1, emb2)


def test_embedder_deterministic_repeated_embedding():
    """Same text should produce identical embeddings on repeated calls."""
    embedder = Embedder()
    text = "def calculate_total(items): return sum(item.price for item in items)"

    emb1 = embedder.embed(text)
    emb2 = embedder.embed(text)

    assert np.allclose(emb1, emb2, atol=1e-6)


def test_embedder_deterministic_across_instances():
    """Embeddings should be deterministic across different Embedder instances."""
    embedder1 = Embedder()
    embedder2 = Embedder()
    text = "def test_function(): return 42"

    emb1 = embedder1.embed(text)
    emb2 = embedder2.embed(text)

    assert np.allclose(emb1, emb2, atol=1e-6)


def test_embedder_model_caching():
    """Multiple Embedder instances should share the same loaded model singleton."""
    embedder1 = Embedder()
    _ = embedder1.embed("test")

    embedder2 = Embedder()
    _ = embedder2.embed("test")

    assert Embedder._model is not None
    assert embedder1._model is embedder2._model


def test_embedder_unit_normalization_single():
    """Single embeddings should be unit vectors (L2 norm ≈ 1.0)."""
    embedder = Embedder()
    text = "def compute_sum(a, b): return a + b"

    embedding = embedder.embed(text)
    norm = np.linalg.norm(embedding)

    assert np.isclose(norm, 1.0, atol=1e-5)


def test_embedder_unit_normalization_batch():
    """All embeddings in a batch should be unit vectors (L2 norm ≈ 1.0)."""
    embedder = Embedder()
    texts = [
        "def add(a, b): return a + b",
        "class Service: pass",
        "import json",
    ]

    embeddings = embedder.embed_batch(texts)
    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedder_custom_model_name():
    """Embedder should accept custom model name parameter."""
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text = "def foo(): pass"

    embedding = embedder.embed(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32


def test_embedder_semantic_similarity_ranking():
    """Semantically related code should produce higher cosine similarity than unrelated code."""
    embedder = Embedder()

    text1 = "def calculate_sum(x, y): return x + y"
    text2 = "def add_numbers(a, b): return a + b"
    text3 = "class DatabaseConnectionPool: pass"

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)
    emb3 = embedder.embed(text3)

    sim_12 = np.dot(emb1, emb2)
    sim_13 = np.dot(emb1, emb3)

    assert sim_12 > sim_13


def test_embedder_batch_smaller_than_batch_size():
    """embed_batch with fewer texts than batch_size should succeed."""
    embedder = Embedder()
    texts = [
        "def hello(): pass",
        "class Foo: pass",
    ]

    embeddings = embedder.embed_batch(texts, batch_size=32)

    assert embeddings.shape == (2, config.EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32


def test_embedder_batch_larger_than_batch_size():
    """embed_batch with more texts than batch_size should process all items across batches."""
    embedder = Embedder()
    texts = [f"def func_{i}(): pass" for i in range(100)]

    embeddings = embedder.embed_batch(texts, batch_size=32)

    assert embeddings.shape == (100, config.EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32


def test_embedder_batch_preserves_input_order():
    """Embeddings in batch should match the exact ordering of the input texts."""
    embedder = Embedder()
    texts = [f"def unique_func_{i}(): return {i * 100}" for i in range(20)]

    batch_embeddings = embedder.embed_batch(texts, batch_size=5)

    assert batch_embeddings.shape == (20, config.EMBEDDING_DIMENSION)
    assert batch_embeddings.dtype == np.float32

    # Verify each row matches the individual embedding of that text
    for i, text in enumerate(texts):
        single_emb = embedder.embed(text)
        assert np.allclose(batch_embeddings[i], single_emb, atol=1e-5)


def test_embedder_batch_custom_batch_size():
    """embed_batch should respect custom batch_size parameter."""
    embedder = Embedder()
    texts = [f"def func_{i}(): pass" for i in range(10)]

    embeddings = embedder.embed_batch(texts, batch_size=3)

    assert embeddings.shape == (10, config.EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32


def test_embedder_batch_generator_mock_invocation():
    """embed_batch should iterate over model.embed generator in correct batch slices."""
    embedder = Embedder()
    # Ensure model is initialized
    _ = embedder.embed("init")

    call_count = 0
    yielded_lengths = []

    def mock_embed(batch):
        nonlocal call_count
        call_count += 1
        yielded_lengths.append(len(batch))
        # Generator yielding 1D float32 numpy arrays
        for _ in range(len(batch)):
            vec = np.ones(config.EMBEDDING_DIMENSION, dtype=np.float32)
            yield vec / np.linalg.norm(vec)

    with patch.object(Embedder._model, "embed", side_effect=mock_embed):
        texts = [f"text_{i}" for i in range(70)]
        embeddings = embedder.embed_batch(texts, batch_size=32)

    # 70 texts with batch_size=32 -> 3 batches (32, 32, 6)
    assert call_count == 3
    assert yielded_lengths == [32, 32, 6]
    assert embeddings.shape == (70, config.EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32


def test_embedder_empty_text():
    """Empty string input should produce a valid 384-dim float32 unit vector."""
    embedder = Embedder()

    embedding = embedder.embed("")

    assert embedding.shape == (config.EMBEDDING_DIMENSION,)
    assert embedding.dtype == np.float32
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_embedder_special_characters():
    """Embedder should handle code strings containing regexes, escape sequences, and unicode."""
    embedder = Embedder()
    text = 'def regex_match(text): return re.match(r"\\d+\\s+🚀", text)'

    embedding = embedder.embed(text)

    assert embedding.shape == (config.EMBEDDING_DIMENSION,)
    assert embedding.dtype == np.float32
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5)
