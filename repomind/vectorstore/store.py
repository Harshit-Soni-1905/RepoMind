"""ChromaDB vector store wrapper for code chunk indexing and semantic retrieval."""

from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import os

from repomind.config import config
from repomind.chunking.models import CodeChunk
from repomind.vectorstore.models import RetrievalResult
from repomind.vectorstore.embedder import Embedder


class VectorStore:
    """ChromaDB vector store for storing code chunks and querying by semantic similarity."""

    def __init__(
        self,
        collection_name: str = "repomind_chunks",
        persist_dir: Optional[Union[str, Path]] = None,
        embedder: Optional[Embedder] = None,
    ):
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_dir: Directory where ChromaDB data will be persisted.
                         If None, uses config.DATA_DIR / "chroma".
                         If string ":memory:", runs in-memory without disk persistence.
            embedder: Embedder instance to generate embeddings.
                     If None, creates a default Embedder().
        """
        import chromadb

        self.collection_name = collection_name
        self.embedder = embedder or Embedder()

        if persist_dir == ":memory:":
            self.client = chromadb.Client()
        else:
            if persist_dir is None:
                persist_path = config.DATA_DIR / "chroma"
            else:
                persist_path = Path(persist_dir)

            os.makedirs(persist_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_path))

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[CodeChunk],
        batch_size: Optional[int] = None,
    ) -> int:
        """Upsert a list of CodeChunk objects into the vector store.

        Generates embeddings and upserts records in small, memory-safe batches
        to avoid holding the full repository embedding matrix in memory at once.

        Deterministic chunk IDs ensure that calling add_chunks repeatedly with
        the same chunks will update (upsert) existing records rather than create duplicates.

        Args:
            chunks: List of CodeChunk objects to index
            batch_size: Maximum number of chunks to process in each embedding/upsert batch.
                       Defaults to config.EMBEDDING_BATCH_SIZE.

        Returns:
            Number of chunks successfully added/upserted
        """
        if not chunks:
            return 0

        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE

        batch_size = max(1, batch_size)

        seen_ids: set = set()
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]

            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            texts_to_embed: List[str] = []

            for chunk in batch_chunks:
                chunk_id = chunk.chunk_id
                if chunk_id in seen_ids:
                    # Disambiguate duplicate ID with start_line if not already present
                    if f"@{chunk.start_line}" not in chunk_id:
                        chunk_id = f"{chunk_id}@{chunk.start_line}"
                    if chunk_id in seen_ids:
                        counter = 2
                        while f"{chunk_id}_{counter}" in seen_ids:
                            counter += 1
                        chunk_id = f"{chunk_id}_{counter}"
                seen_ids.add(chunk_id)

                ids.append(chunk_id)
                documents.append(chunk.source_code)
                texts_to_embed.append(chunk.source_code)

                # Metadata in ChromaDB must be primitive types: str, int, float, bool
                meta: Dict[str, Any] = {
                    "filepath": chunk.filepath.as_posix(),
                    "chunk_type": chunk.chunk_type,
                    "symbol_name": chunk.symbol_name or "",
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "docstring": chunk.docstring or "",
                    "parent_class": chunk.parent_class or "",
                    "decorators": ",".join(chunk.decorators) if chunk.decorators else "",
                }
                metadatas.append(meta)

            # Generate embeddings for this batch only
            embeddings = self.embedder.embed_batch(texts_to_embed, batch_size=batch_size)
            embeddings_list = embeddings.tolist()

            # ChromaDB upsert for this batch only
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )

            total_added += len(batch_chunks)

        return total_added

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform semantic similarity search for a text query.

        Args:
            query: Natural language or code query text
            top_k: Maximum number of results to return.
                   Defaults to config.VECTOR_SEARCH_TOP_K.
            where: Optional ChromaDB metadata filter dictionary (e.g. {"chunk_type": "function"})

        Returns:
            List of RetrievalResult objects sorted by similarity score (descending)
        """
        if not query or not query.strip():
            return []

        # Return empty list if vector store contains no documents
        if self.count() == 0:
            return []

        k = top_k if top_k is not None else config.VECTOR_SEARCH_TOP_K
        k = max(1, k)

        # Generate query embedding
        query_embedding = self.embedder.embed(query).tolist()

        # Perform query in ChromaDB
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        retrieval_results: List[RetrievalResult] = []

        retrieved_ids = results["ids"][0]
        retrieved_documents = results["documents"][0]
        retrieved_metadatas = results["metadatas"][0]
        retrieved_distances = results["distances"][0]

        for i, (chunk_id, doc, meta, distance) in enumerate(
            zip(retrieved_ids, retrieved_documents, retrieved_metadatas, retrieved_distances)
        ):
            # Convert metadata back to CodeChunk instance
            filepath = Path(meta.get("filepath", ""))
            chunk_type = meta.get("chunk_type", "file")
            symbol_name = meta.get("symbol_name") or None
            start_line = int(meta.get("start_line", 1))
            end_line = int(meta.get("end_line", 1))
            docstring = meta.get("docstring") or None
            parent_class = meta.get("parent_class") or None
            decorators_str = meta.get("decorators", "")
            decorators = decorators_str.split(",") if decorators_str else []

            chunk = CodeChunk(
                chunk_id=chunk_id,
                filepath=filepath,
                chunk_type=chunk_type,
                symbol_name=symbol_name,
                source_code=doc,
                start_line=start_line,
                end_line=end_line,
                docstring=docstring,
                parent_class=parent_class,
                decorators=decorators,
            )

            # Cosine distance ranges [0, 2]. Cosine similarity = 1.0 - distance
            # Clamp similarity score to range [-1.0, 1.0]
            similarity_score = 1.0 - distance

            retrieval_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=similarity_score,
                    distance=distance,
                    rank=i + 1,
                )
            )

        return retrieval_results

    def count(self) -> int:
        """Get total number of chunks currently stored in the vector store.

        Returns:
            Integer count of stored chunks
        """
        return self.collection.count()

    def clear(self) -> None:
        """Delete all chunks from the current collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
