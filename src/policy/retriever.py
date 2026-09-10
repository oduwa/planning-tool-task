"""In-memory semantic search over the persisted policy index.

Beyond a plain top-k cosine lookup, this retriever supports the two things the
agent needs to investigate policy like an officer rather than in a single pass:

- ``search`` with **Maximal Marginal Relevance (MMR)** so repeated queries on a
  theme surface *diverse* passages (different policies/documents) instead of near
  duplicates, which is what makes iterative deepening actually add evidence.
- ``lookup_reference`` to pull every passage that mentions a specific named policy
  or paragraph. The agent uses it to read a policy in full before relying on it,
  and the grounding guardrail uses it to verify that a cited policy exists in the
  corpus at all (catching hallucinated citations).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from llm import ModelConfig
from paths import POLICY_INDEX_DIR
from policy.embedder import Embedder
from policy.index import CHUNKS_FILE, VECTORS_FILE, PolicyChunk

# Pool of top-similarity candidates that MMR re-ranks for diversity.
_MMR_POOL = 40


class PolicyRetriever:
    """Load a policy index and answer diverse, citation-checkable queries.

    Vectors are L2-normalized at build time, so cosine similarity against the
    corpus is a single matrix-vector dot product.
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

    @property
    def source_files(self) -> set[str]:
        """Return the set of distinct policy document names in the index."""
        return {chunk.source_file for chunk in self._chunks}

    def search(self, query: str, k: int = 6, *, mmr_lambda: float = 0.6) -> list[PolicyChunk]:
        """Return the ``k`` most relevant *and diverse* chunks for ``query``.

        Args:
            query: Free-text query, typically a planning theme + case facts.
            k: Number of chunks to return.
            mmr_lambda: Relevance/diversity trade-off in [0, 1]; higher favours
                relevance, lower favours diversity.
        """
        if not self._chunks:
            return []
        query_vector = self._embedder.embed_one(query)
        scores = self._vectors @ query_vector

        pool_size = min(_MMR_POOL, len(self._chunks))
        pool = np.argsort(scores)[::-1][:pool_size].tolist()

        selected: list[int] = []
        while pool and len(selected) < k:
            if not selected:
                best = max(pool, key=lambda i: scores[i])
            else:
                selected_matrix = self._vectors[selected]

                def _mmr_score(i: int, sel: NDArray[np.float32] = selected_matrix) -> float:
                    redundancy = float(np.max(sel @ self._vectors[i]))
                    return mmr_lambda * float(scores[i]) - (1.0 - mmr_lambda) * redundancy

                best = max(pool, key=_mmr_score)
            selected.append(best)
            pool.remove(best)
        return [self._chunks[i] for i in selected]

    def lookup_reference(self, reference: str, k: int = 6) -> list[PolicyChunk]:
        """Return chunks that mention a specific named policy or paragraph.

        Matches either the detected ``policy_ref`` tag or the reference phrase in
        the chunk text, tolerating spacing/case vari/'para' vs 'paragraph'.

        Args:
            reference: A citation such as ``"Policy 44"`` or ``"NPPF paragraph 135"``.
            k: Maximum number of chunks to return.
        """
        pattern = self._reference_pattern(reference)
        if pattern is None:
            return []
        hits = [
            chunk
            for chunk in self._chunks
            if (chunk.policy_ref and pattern.search(chunk.policy_ref))
            or pattern.search(chunk.text)
        ]
        return hits[:k]

    def verify_citation(self, reference: str) -> bool:
        """Return True if the corpus contains any passage matching ``reference``.

        Args:
            reference: A citation such as ``"Policy 13"`` or ``"paragraph 135"``.
        """
        return bool(self.lookup_reference(reference, k=1))

    @staticmethod
    def _reference_pattern(reference: str) -> re.Pattern[str] | None:
        """Build a tolerant regex for a policy/paragraph citation.

        Args:
            reference: The raw citation text.
        """
        match = re.search(
            r"(policy)\s+([0-9]+[a-z]?)|(?:paragraph|para\.?)\s+([0-9]+)",
            reference,
            re.IGNORECASE,
        )
        if not match:
            return None
        if match.group(1):
            return re.compile(rf"\bpolicy\s+{match.group(2)}\b", re.IGNORECASE)
        number = match.group(3)
        return re.compile(rf"\b(?:paragraph|para\.?)\s+{number}\b", re.IGNORECASE)

    @staticmethod
    def format_results(chunks: list[PolicyChunk]) -> str:
        """Render retrieved chunks as a compact, citation-friendly string.

        Args:
            chunks: Chunks returned by :meth:`search` or :meth:`lookup_reference`.
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
