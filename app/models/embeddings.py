from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wraps a SentenceTransformer model for batch encoding of texts.

    Design goals:
    - Use a small but effective model (default: all-MiniLM-L6-v2).
    - Support batch encoding for speed and efficiency.
    - Normalize embeddings for cosine similarity consistency.

    Typical usage:
    embedder = EmbeddingModel()
    vectors = embedder.encode_batch(["hello world", "puppies are cute"])
    print(vectors.shape) # (2, 384)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Convert a list of texts to embeddings.

        Args:
            texts: List of input strings.
            normalize: Whether to L2-normalize embeddings (default: True).

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings.astype(np.float32)

__all__ = ["EmbeddingModel"]