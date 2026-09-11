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
