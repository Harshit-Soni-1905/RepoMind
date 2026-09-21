"""Tests for ChromaDB vector store."""

import pytest
from pathlib import Path
import numpy as np

from repomind.chunking.models import CodeChunk
from repomind.vectorstore.store import VectorStore
from repomind.vectorstore.models import RetrievalResult
from repomind.vectorstore.embedder import Embedder


@pytest.fixture
def sample_chunks():
    """Create sample code chunks for testing."""
    return [
        CodeChunk(
            chunk_id="test.py::add",
            filepath=Path("test.py"),
            chunk_type="function",
            symbol_name="add",
            source_code="def add(a, b):\n    return a + b",
            start_line=1,
            end_line=2,
            docstring="Add two numbers.",
        ),
        CodeChunk(
            chunk_id="test.py::multiply",
            filepath=Path("test.py"),
            chunk_type="function",
            symbol_name="multiply",
            source_code="def multiply(x, y):\n    return x * y",
            start_line=4,
            end_line=5,
            docstring="Multiply two numbers.",
        ),
        CodeChunk(
            chunk_id="utils.py::Database",
            filepath=Path("utils.py"),
            chunk_type="class",
            symbol_name="Database",
            source_code="class Database:\n    def connect(self):\n        pass",
            start_line=1,
            end_line=3,
            docstring="Database connection class.",
        ),
    ]


@pytest.fixture
def in_memory_store():
    """Create an in-memory vector store for testing with unique collection name."""
    import uuid
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    return VectorStore(collection_name=collection_name, persist_dir=":memory:")


def test_vectorstore_add_chunks(in_memory_store, sample_chunks):
    """VectorStore should successfully add chunks."""
    count = in_memory_store.add_chunks(sample_chunks)

    assert count == 3
    assert in_memory_store.count() == 3


def test_vectorstore_add_empty_list(in_memory_store):
    """VectorStore should handle empty chunk list."""
    count = in_memory_store.add_chunks([])

    assert count == 0
    assert in_memory_store.count() == 0


def test_vectorstore_upsert_behavior(in_memory_store, sample_chunks):
    """Re-adding same chunks should update (upsert), not duplicate."""
    # Add chunks first time
    in_memory_store.add_chunks(sample_chunks)
    assert in_memory_store.count() == 3

    # Add same chunks again
    in_memory_store.add_chunks(sample_chunks)

    # Should still have 3 chunks, not 6
    assert in_memory_store.count() == 3


def test_vectorstore_search_basic(in_memory_store, sample_chunks):
    """VectorStore should return relevant results for semantic search."""
    in_memory_store.add_chunks(sample_chunks)

    results = in_memory_store.search("function that adds two numbers", top_k=5)

    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    # Most relevant should be the 'add' function
    assert results[0].chunk.symbol_name == "add"


def test_vectorstore_search_top_k(in_memory_store, sample_chunks):
    """VectorStore should respect top_k parameter."""
    in_memory_store.add_chunks(sample_chunks)

    results = in_memory_store.search("mathematical function", top_k=2)

    assert len(results) == 2


def test_vectorstore_search_empty_store(in_memory_store):
    """Search on empty store should return empty list."""
    results = in_memory_store.search("some query", top_k=5)

    assert results == []


def test_vectorstore_search_empty_query(in_memory_store, sample_chunks):
    """Empty query should return empty list."""
    in_memory_store.add_chunks(sample_chunks)

    results = in_memory_store.search("", top_k=5)
    assert results == []

    results = in_memory_store.search("   ", top_k=5)
    assert results == []


def test_vectorstore_retrieval_result_structure(in_memory_store, sample_chunks):
    """RetrievalResult should have correct structure and metadata."""
    in_memory_store.add_chunks(sample_chunks)

    results = in_memory_store.search("add numbers", top_k=1)

    assert len(results) == 1
    result = results[0]

    # Check RetrievalResult fields
    assert isinstance(result.chunk, CodeChunk)
    assert isinstance(result.score, float)
    assert isinstance(result.distance, float)
    assert isinstance(result.rank, int)
    assert result.rank == 1

    # Score should be in reasonable range (cosine similarity: -1 to 1)
    assert -1.0 <= result.score <= 1.0

    # Distance should be non-negative
    assert result.distance >= 0.0


def test_vectorstore_metadata_preservation(in_memory_store, sample_chunks):
    """Metadata should be preserved through indexing and retrieval."""
    chunk = sample_chunks[0]  # add function
    in_memory_store.add_chunks([chunk])

    results = in_memory_store.search("addition function", top_k=1)
    retrieved_chunk = results[0].chunk

    assert retrieved_chunk.chunk_id == chunk.chunk_id
    assert retrieved_chunk.filepath == chunk.filepath
    assert retrieved_chunk.chunk_type == chunk.chunk_type
    assert retrieved_chunk.symbol_name == chunk.symbol_name
    assert retrieved_chunk.source_code == chunk.source_code
    assert retrieved_chunk.start_line == chunk.start_line
    assert retrieved_chunk.end_line == chunk.end_line
    assert retrieved_chunk.docstring == chunk.docstring


def test_vectorstore_chunk_with_parent_class(in_memory_store):
    """Chunks with parent_class and decorators should be stored correctly."""
    chunk = CodeChunk(
        chunk_id="models.py::User.save",
        filepath=Path("models.py"),
        chunk_type="method",
        symbol_name="save",
        source_code="@property\ndef save(self):\n    pass",
        start_line=10,
        end_line=12,
        parent_class="User",
        decorators=["@property"],
    )

    in_memory_store.add_chunks([chunk])
    results = in_memory_store.search("save method", top_k=1)

    retrieved = results[0].chunk
    assert retrieved.parent_class == "User"
    assert retrieved.decorators == ["@property"]


def test_vectorstore_results_ranked(in_memory_store, sample_chunks):
    """Results should be ranked by relevance."""
    in_memory_store.add_chunks(sample_chunks)

    results = in_memory_store.search("arithmetic operations", top_k=3)

    # Ranks should be sequential starting from 1
    ranks = [r.rank for r in results]
    assert ranks == list(range(1, len(results) + 1))

    # Scores should be in descending order (most relevant first)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_vectorstore_count(in_memory_store, sample_chunks):
    """Count should return correct number of stored chunks."""
    assert in_memory_store.count() == 0

    in_memory_store.add_chunks([sample_chunks[0]])
    assert in_memory_store.count() == 1

    in_memory_store.add_chunks(sample_chunks[1:])
    assert in_memory_store.count() == 3


def test_vectorstore_clear(in_memory_store, sample_chunks):
    """Clear should remove all chunks from the store."""
    in_memory_store.add_chunks(sample_chunks)
    assert in_memory_store.count() == 3

    in_memory_store.clear()

    assert in_memory_store.count() == 0
    results = in_memory_store.search("any query", top_k=5)
    assert results == []


def test_vectorstore_metadata_filter(in_memory_store, sample_chunks):
    """VectorStore should support metadata filtering."""
    in_memory_store.add_chunks(sample_chunks)

    # Search only for functions
    results = in_memory_store.search(
        "code",
        top_k=10,
        where={"chunk_type": "function"}
    )

    # Should only return functions, not the class
    assert len(results) == 2
    assert all(r.chunk.chunk_type == "function" for r in results)


def test_vectorstore_custom_embedder(sample_chunks):
    """VectorStore should accept custom embedder instance."""
    custom_embedder = Embedder()
    store = VectorStore(
        collection_name="custom_test",
        persist_dir=":memory:",
        embedder=custom_embedder
    )

    store.add_chunks(sample_chunks)
    results = store.search("add function", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.symbol_name == "add"


def test_vectorstore_deterministic_chunk_ids(in_memory_store):
    """Chunks with same ID should update rather than duplicate."""
    chunk1 = CodeChunk(
        chunk_id="test.py::foo",
        filepath=Path("test.py"),
        chunk_type="function",
        symbol_name="foo",
        source_code="def foo(): return 1",
        start_line=1,
        end_line=1,
    )

    chunk2 = CodeChunk(
        chunk_id="test.py::foo",  # Same ID
        filepath=Path("test.py"),
        chunk_type="function",
        symbol_name="foo",
        source_code="def foo(): return 2",  # Different content
        start_line=1,
        end_line=1,
    )

    in_memory_store.add_chunks([chunk1])
    assert in_memory_store.count() == 1

    in_memory_store.add_chunks([chunk2])
    assert in_memory_store.count() == 1  # Still 1, not 2

    # Retrieve and verify it's the updated version
    results = in_memory_store.search("foo function", top_k=1)
    assert "return 2" in results[0].chunk.source_code


def test_vectorstore_semantic_similarity_ordering(in_memory_store):
    """Results should be ordered by semantic similarity to query."""
    chunks = [
        CodeChunk(
            chunk_id="math.py::add",
            filepath=Path("math.py"),
            chunk_type="function",
            symbol_name="add",
            source_code="def add(a, b): return a + b",
            start_line=1,
            end_line=1,
            docstring="Add two numbers together.",
        ),
        CodeChunk(
            chunk_id="math.py::subtract",
            filepath=Path("math.py"),
            chunk_type="function",
            symbol_name="subtract",
            source_code="def subtract(a, b): return a - b",
            start_line=3,
            end_line=3,
            docstring="Subtract one number from another.",
        ),
        CodeChunk(
            chunk_id="io.py::read_file",
            filepath=Path("io.py"),
            chunk_type="function",
            symbol_name="read_file",
            source_code="def read_file(path): return open(path).read()",
            start_line=1,
            end_line=1,
            docstring="Read contents of a file.",
        ),
    ]

    in_memory_store.add_chunks(chunks)

    # Query for addition - should rank 'add' highest
    results = in_memory_store.search("function to add two numbers", top_k=3)
    assert results[0].chunk.symbol_name == "add"

    # Query for file reading - should rank 'read_file' highest
    results = in_memory_store.search("read file contents", top_k=3)
    assert results[0].chunk.symbol_name == "read_file"


def test_vectorstore_integration_real_embeddings(in_memory_store):
    """Integration test with real sentence-transformers model."""
    # This test uses the actual model, not mocked
    chunk = CodeChunk(
        chunk_id="example.py::factorial",
        filepath=Path("example.py"),
        chunk_type="function",
        symbol_name="factorial",
        source_code="""def factorial(n):
    '''Calculate factorial of n.'''
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
        start_line=1,
        end_line=5,
        docstring="Calculate factorial of n.",
    )

    in_memory_store.add_chunks([chunk])
    results = in_memory_store.search("recursive factorial computation", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.symbol_name == "factorial"
    assert results[0].score > 0.0  # Should have positive similarity


def test_vectorstore_handles_special_characters(in_memory_store):
    """VectorStore should handle code with special characters."""
    chunk = CodeChunk(
        chunk_id="test.py::regex_match",
        filepath=Path("test.py"),
        chunk_type="function",
        symbol_name="regex_match",
        source_code='def regex_match(text): return re.match(r"\\d+", text)',
        start_line=1,
        end_line=1,
    )

    count = in_memory_store.add_chunks([chunk])
    assert count == 1

    results = in_memory_store.search("pattern matching", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.symbol_name == "regex_match"


# ---- Incremental Batching Tests ----

def test_vectorstore_add_chunks_multi_batch(in_memory_store):
    """VectorStore should successfully index chunks when processed across multiple batches."""
    chunks = [
        CodeChunk(
            chunk_id=f"func_{i}.py::func_{i}",
            filepath=Path(f"func_{i}.py"),
            chunk_type="function",
            symbol_name=f"func_{i}",
            source_code=f"def func_{i}():\n    return {i}",
            start_line=1,
            end_line=2,
            docstring=f"Function number {i}",
        )
        for i in range(10)
    ]

    # Process 10 chunks in batches of 3 (4 batches: 3, 3, 3, 1)
    count = in_memory_store.add_chunks(chunks, batch_size=3)

    assert count == 10
    assert in_memory_store.count() == 10

    # Verify search retrieves items across batches
    results = in_memory_store.search("function return", top_k=10)
    assert len(results) == 10


def test_vectorstore_add_chunks_batch_size_one(in_memory_store, sample_chunks):
    """VectorStore should handle batch_size=1 without issues."""
    count = in_memory_store.add_chunks(sample_chunks, batch_size=1)

    assert count == len(sample_chunks)
    assert in_memory_store.count() == len(sample_chunks)


def test_vectorstore_add_chunks_batch_size_larger_than_total(in_memory_store, sample_chunks):
    """VectorStore should handle batch_size greater than total chunks."""
    count = in_memory_store.add_chunks(sample_chunks, batch_size=100)

    assert count == len(sample_chunks)
    assert in_memory_store.count() == len(sample_chunks)


def test_vectorstore_add_chunks_incremental_embed_and_upsert_calls():
    """VectorStore should call embed_batch and collection.upsert per batch slice."""
    from unittest.mock import MagicMock, patch

    mock_embedder = MagicMock(spec=Embedder)
    # Return dummy numpy embedding array for any batch
    mock_embedder.embed_batch.side_effect = lambda texts, batch_size=None: np.zeros((len(texts), 384))

    store = VectorStore(
        collection_name="mock_batch_test",
        persist_dir=":memory:",
        embedder=mock_embedder,
    )

    # Mock collection.upsert
    store.collection.upsert = MagicMock()

    chunks = [
        CodeChunk(
            chunk_id=f"item_{i}.py::item_{i}",
            filepath=Path(f"item_{i}.py"),
            chunk_type="function",
            symbol_name=f"item_{i}",
            source_code=f"def item_{i}(): pass",
            start_line=1,
            end_line=1,
        )
        for i in range(25)
    ]

    # 25 chunks with batch_size=10 -> 3 batches (10, 10, 5)
    count = store.add_chunks(chunks, batch_size=10)

    assert count == 25
    assert mock_embedder.embed_batch.call_count == 3
    assert store.collection.upsert.call_count == 3

    # Check batch sizes in embedder calls
    embed_calls = mock_embedder.embed_batch.call_args_list
    assert len(embed_calls[0][0][0]) == 10
    assert len(embed_calls[1][0][0]) == 10
    assert len(embed_calls[2][0][0]) == 5

    # Check batch sizes in upsert calls
    upsert_calls = store.collection.upsert.call_args_list
    assert len(upsert_calls[0][1]["ids"]) == 10
    assert len(upsert_calls[1][1]["ids"]) == 10
    assert len(upsert_calls[2][1]["ids"]) == 5


def test_vectorstore_add_chunks_duplicate_disambiguation_across_batches(in_memory_store):
    """Duplicate chunk IDs appearing across different batches should be properly disambiguated."""
    # 6 chunks with identical base chunk_id across multiple batches (batch_size=2 -> 3 batches)
    chunks = [
        CodeChunk(
            chunk_id="collision.py::foo",
            filepath=Path("collision.py"),
            chunk_type="function",
            symbol_name="foo",
            source_code=f"def foo(): return {i}",
            start_line=i * 10 + 1,
            end_line=i * 10 + 5,
        )
        for i in range(6)
    ]

    count = in_memory_store.add_chunks(chunks, batch_size=2)

    assert count == 6
    assert in_memory_store.count() == 6


def test_vectorstore_preserves_metadata_across_batches(in_memory_store):
    """Metadata should be preserved for all chunks when indexed across multiple batches."""
    chunks = [
        CodeChunk(
            chunk_id=f"service_{i}.py::Service{i}.run",
            filepath=Path(f"service_{i}.py"),
            chunk_type="method",
            symbol_name="run",
            source_code=f"@logger\ndef run(self):\n    '''Run service {i}'''\n    return {i}",
            start_line=10,
            end_line=15,
            docstring=f"Run service {i}",
            parent_class=f"Service{i}",
            decorators=["@logger"],
        )
        for i in range(6)
    ]

    in_memory_store.add_chunks(chunks, batch_size=2)

    results = in_memory_store.search("Run service 3", top_k=6)
    assert len(results) == 6

    # Verify metadata fields are preserved
    for r in results:
        assert r.chunk.parent_class is not None
        assert r.chunk.parent_class.startswith("Service")
        assert r.chunk.decorators == ["@logger"]
        assert r.chunk.start_line == 10
        assert r.chunk.end_line == 15

