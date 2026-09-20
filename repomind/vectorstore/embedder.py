"""Embedding generator for code chunks.

This module provides a wrapper around sentence-transformers to generate
embeddings for code chunks. The model is loaded lazily and cached to avoid
reloading on every call.
"""

from typing import List, Union
import numpy as np

from repomind.config import config


class Embedder:
    """Generate embeddings for text using sentence-transformers.

    The model is loaded lazily on first use and cached as a class variable
    to avoid reloading for every instance.
    """

    _model = None  # Class-level cache for the loaded model
    _model_name = None  # Track which model is loaded

    def __init__(self, model_name: str = None):
        """Initialize embedder.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to config.EMBEDDING_MODEL.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL

    def _load_model(self):
        """Load the sentence-transformers model if not already loaded."""
        # Check if we need to load or reload the model
        if Embedder._model is None or Embedder._model_name != self.model_name:
            from sentence_transformers import SentenceTransformer
            Embedder._model = SentenceTransformer(self.model_name)
            Embedder._model_name = self.model_name

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            Numpy array of shape (embedding_dim,) containing the embedding vector
        """
        self._load_model()
        # encode() returns ndarray of shape (1, embedding_dim) for single string
        # We squeeze to get (embedding_dim,)
        embedding = Embedder._model.encode([text], convert_to_numpy=True)
        return embedding[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = None,
    ) -> np.ndarray:
        """Generate embeddings for a batch of text strings with memory-safe batching.

        Processes texts in smaller batches to control peak memory usage,
        which is critical for constrained environments like Render free tier.

        Args:
            texts: List of texts to embed
            batch_size: Maximum number of texts to process at once.
                       Defaults to config.EMBEDDING_BATCH_SIZE.

        Returns:
            Numpy array of shape (num_texts, embedding_dim) containing embedding vectors
            in the same order as input texts.
        """
        if not texts:
            return np.array([])

        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE

        self._load_model()
        all_embeddings = []

        # Process in batches to limit memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = Embedder._model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(embeddings)

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])

    @property
    def dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        self._load_model()
        # Get dimension from model's config
        return Embedder._model.get_embedding_dimension()
