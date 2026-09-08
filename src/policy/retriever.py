"""In-memory cosine-similarity search over the persisted policy index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from llm import ModelConfig
from paths import POLICY_INDEX_DIR
from policy.embedder import Embedder
from policy.index import CHUNKS_FILE, VECTORS_FILE, PolicyChunk


class PolicyRetriever:
    """Load a policy index and answer top-k semantic queries.

    Vectors are already L2-normalized at build time, so a query's cosine
    similarity against the corpus is a single matrix-vector dot product.
    """

    def __init__(self, cfg: ModelConfig, index_dir: Path = POLICY_INDEX_DIR) -> None:
        """Load the index artifacts from disk.

        Args:
            cfg: Model config (supplies the embedding model name).
            index_dir: Directory containing the index artifacts.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        vectors_path = index_dir / VECTORS_FILE
        chunks_path = index_dir / CHUNKS_FILE
        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Policy index not found in {index_dir}. Build it first with "
                "`uv run planning build-index`."
            )
        self._vectors: NDArray[np.float32] = np.load(vectors_path)
        with chunks_path.open(encoding="utf-8") as handle:
            self._chunks = [PolicyChunk.model_validate_json(line) for line in handle]
        self._embedder = Embedder(cfg.embed_model)

    def search(self, query: str, k: int = 6) -> list[PolicyChunk]:
        """Return the ``k`` most similar policy chunks to ``query``.

        Args:
            query: Free-text query, typically a planning theme + case facts.
            k: Number of chunks to return.
        """
        if not self._chunks:
            return []
        query_vector = self._embedder.embed_one(query)
        scores = self._vectors @ query_vector
        top = np.argsort(scores)[::-1][:k]
        return [self._chunks[int(i)] for i in top]

    @staticmethod
    def format_results(chunks: list[PolicyChunk]) -> str:
        """Render retrieved chunks as a compact, citation-friendly string.

        Args:
            chunks: Chunks returned by :meth:`search`.
        """
        if not chunks:
            return "No relevant policy passages found."
        lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            ref = f" [{chunk.policy_ref}]" if chunk.policy_ref else ""
            lines.append(
                f"{index}. ({chunk.source_file}, p.{chunk.page}){ref}\n{chunk.text}"
            )
        return "\n\n".join(lines)
