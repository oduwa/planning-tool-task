"""Local text embeddings via fastembed.

A local model keeps retrieval free and offline. OpenRouter is a chat gateway and
does not reliably expose an embeddings endpoint, so embeddings never leave the
machine. Vectors are L2-normalized so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from numpy.typing import NDArray


class Embedder:
    """Wrapper around a fastembed text-embedding model."""

    def __init__(self, model_name: str) -> None:
        """Initialize the embedding model.

        Args:
            model_name: A fastembed-supported model id, e.g. ``BAAI/bge-small-en-v1.5``.

        Raises:
            RuntimeError: If fastembed is not installed.
        """
        try:
            from fastembed import TextEmbedding
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "fastembed is required for policy retrieval. Install it with "
                "`uv add fastembed`."
            ) from error
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int = 128,
        log_every: int = 0,
    ) -> NDArray[np.float32]:
        """Embed a list of texts into an L2-normalized matrix.

        Args:
            texts: Texts to embed.
            batch_size: Number of texts handed to the model per step.
            log_every: If > 0, log progress after every this many texts.

        Returns:
            A ``(len(texts), dim)`` float32 array of unit vectors.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # parallel=None runs in-process (onnxruntime threading). Data-parallel
        # worker pools (parallel>=1) crash silently in this sandbox, so avoid them.
        vectors_iter = self._model.embed(texts, batch_size=batch_size, parallel=None)
        rows: list[NDArray[np.float32]] = []
        for row in vectors_iter:
            rows.append(np.asarray(row, dtype=np.float32))
            if log_every and len(rows) % log_every == 0:
                logger.info(f"Embedded {len(rows)}/{len(texts)} texts")
        vectors = np.vstack(rows).astype(np.float32)
        return _l2_normalize(vectors)

    def embed_one(self, text: str) -> NDArray[np.float32]:
        """Embed a single text into a normalized 1-D vector.

        Args:
            text: Text to embed.
        """
        vector: NDArray[np.float32] = self.embed([text])[0]
        return vector


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return row-wise L2-normalized vectors, guarding against zero norms."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: NDArray[np.float32] = (matrix / norms).astype(np.float32)
    return normalized
