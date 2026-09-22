"""Embedding generation using FastEmbed (ONNX Runtime) for memory-efficient local embeddings."""

import os
from typing import List, Optional

import numpy as np

from repomind.config import config
from repomind.utils import log_memory


class Embedder:
    """Generate text embeddings using FastEmbed (ONNX Runtime) for low memory footprint.

    This class maintains a class-level singleton cache for the embedding model
    to avoid reloading the model multiple times across the application lifecycle.

    Attributes:
        _model: Class-level cached TextEmbedding instance.
        _model_name: Identifier of the currently loaded model.
    """

    _model = None
    _model_name = None

    def __init__(self, model_name: Optional[str] = None):
        """Initialize Embedder with optional model override.

        Args:
            model_name: Optional model identifier. Defaults to config.EMBEDDING_MODEL.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL

    def _load_model(self) -> None:
        """Lazy-load the FastEmbed model with class-level caching.

        Initializes the model only once per model_name. Thread-safe for single-threaded use.
        """
        if Embedder._model is None or Embedder._model_name != self.model_name:
            from fastembed import TextEmbedding

            # Configure ONNX Runtime thread pool via environment variable before init
            onnx_threads = os.environ.get("REPOMIND_ONNX_THREADS")
            if onnx_threads:
                os.environ["OMP_NUM_THREADS"] = onnx_threads
                os.environ["ONNX_RUNTIME_NUM_THREADS"] = onnx_threads

            log_memory("BEFORE_MODEL_INIT", f"model={self.model_name}")
            Embedder._model = TextEmbedding(model_name=self.model_name)
            Embedder._model_name = self.model_name
            log_memory("MODEL_LOADED", f"model={self.model_name}")

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string.

        Args:
            text: Text to embed.

        Returns:
            Numpy array of shape (384,) with dtype float32 containing the embedding vector.
        """
        self._load_model()
        # FastEmbed embed() returns a generator yielding 1D numpy arrays
        embedding_gen = Embedder._model.embed([text])
        embedding = next(embedding_gen)
        return np.asarray(embedding, dtype=np.float32)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate embeddings for a batch of text strings with memory-safe batching.

        Processes texts in smaller batches to control peak memory usage,
        which is critical for constrained environments like Render free tier.

        Args:
            texts: List of texts to embed.
            batch_size: Maximum number of texts to process at once.
                       Defaults to config.EMBEDDING_BATCH_SIZE.

        Returns:
            Numpy array of shape (num_texts, 384) with dtype float32
            containing embedding vectors in the same order as input texts.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE

        self._load_model()
        all_embeddings: List[np.ndarray] = []

        # Process in batches to limit memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = list(Embedder._model.embed(batch))
            batch_array = np.asarray(batch_embeddings, dtype=np.float32)
            all_embeddings.append(batch_array)

        return np.vstack(all_embeddings)

    @property
    def dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this model.

        Returns:
            Embedding dimension (384 for all-MiniLM-L6-v2).
        """
        if self.model_name == config.EMBEDDING_MODEL or self.model_name == "sentence-transformers/all-MiniLM-L6-v2":
            return config.EMBEDDING_DIMENSION
        self._load_model()
        test_embedding = next(Embedder._model.embed(["test"]))
        return len(test_embedding)
