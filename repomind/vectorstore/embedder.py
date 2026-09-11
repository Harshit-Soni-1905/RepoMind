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

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of text strings.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of shape (num_texts, embedding_dim) containing embedding vectors
        """
        if not texts:
            return np.array([])

        self._load_model()
        # encode() returns ndarray of shape (num_texts, embedding_dim)
        embeddings = Embedder._model.encode(texts, convert_to_numpy=True)
        return embeddings

    @property
    def dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        self._load_model()
        # Get dimension from model's config
        return Embedder._model.get_embedding_dimension()
