"""Build and persist the policy embedding index (one-time, cached to disk).

Every PDF under the national/local policy directories is parsed to text, chunked
into overlapping windows tagged with any detectable policy reference, embedded
with a local model, and written to ``.policy_index/`` as a numpy matrix plus a
JSONL sidecar of chunk metadata.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
from kreuzberg import (  # type: ignore[attr-defined]
    ExtractionConfig,
    ImageExtractionConfig,
    PageConfig,
    PdfConfig,
    ResultFormat,
)
from loguru import logger
from pydantic import BaseModel

from llm import ModelConfig
from parser.parser import extract_pdf_document
from paths import LOCAL_POLICY_DIR, NATIONAL_POLICY_DIR, POLICY_INDEX_DIR, SPDS_DIR
from policy.embedder import Embedder

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"

WORDS_PER_CHUNK = 220
WORD_OVERLAP = 40

_POLICY_REF_PATTERNS = (
    re.compile(r"\bPolicy\s+\d+[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\b(?:NPPF\s+)?paragraph\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bpara\.?\s+\d+\b", re.IGNORECASE),
)


class PolicyChunk(BaseModel):
    """A retrievable passage of policy text with provenance."""

    text: str
    source_file: str
    page: int
    policy_ref: str | None = None


def _text_only_config() -> ExtractionConfig:
    """Build a text-only extraction config for policy PDFs."""
    return ExtractionConfig(
        result_format=ResultFormat.ELEMENT_BASED,
        pages=PageConfig(extract_pages=True),
        images=ImageExtractionConfig(extract_images=False),
        pdf_options=PdfConfig(extract_metadata=True),
    )


def _detect_policy_ref(text: str) -> str | None:
    """Return the first detectable policy/paragraph reference in the text.

    Args:
        text: Chunk text to scan.
    """
    for pattern in _POLICY_REF_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return None


def _chunk_page(text: str) -> list[str]:
    """Split page text into overlapping word windows.

    Args:
        text: Page text.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = WORDS_PER_CHUNK - WORD_OVERLAP
    for start in range(0, len(words), step):
        window = words[start : start + WORDS_PER_CHUNK]
        if window:
            chunks.append(" ".join(window))
        if start + WORDS_PER_CHUNK >= len(words):
            break
    return chunks


async def _chunk_pdf(path: Path) -> list[PolicyChunk]:
    """Parse and chunk a single policy PDF.

    Args:
        path: Path to the policy PDF.
    """
    try:
        document = await extract_pdf_document(path, config=_text_only_config())
    except Exception as error:  # noqa: BLE001 - keep indexing resilient
        logger.warning(f"Skipping {path.name}: extraction failed ({error})")
        return []

    chunks: list[PolicyChunk] = []
    for page in document.pages:
        for window in _chunk_page(page.text):
            chunks.append(
                PolicyChunk(
                    text=window,
                    source_file=path.name,
                    page=page.page_number,
                    policy_ref=_detect_policy_ref(window),
                )
            )
    return chunks


def _policy_pdfs() -> list[Path]:
    """Return every policy PDF across national, local, and SPD directories."""
    directories = [NATIONAL_POLICY_DIR, LOCAL_POLICY_DIR, SPDS_DIR]
    seen: set[Path] = set()
    pdfs: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.pdf")):
            if path not in seen:
                seen.add(path)
                pdfs.append(path)
    return pdfs


async def build_policy_index(
    cfg: ModelConfig, index_dir: Path = POLICY_INDEX_DIR, *, force: bool = False
) -> int:
    """Build the policy embedding index, caching it to ``index_dir``.

    Args:
        cfg: Model config (supplies the embedding model name).
        index_dir: Output directory for the index artifacts.
        force: Rebuild even if an index already exists.

    Returns:
        The number of chunks indexed.
    """
    vectors_path = index_dir / VECTORS_FILE
    if vectors_path.exists() and not force:
        existing = int(np.load(vectors_path, mmap_mode="r").shape[0])
        logger.info(f"Policy index already exists ({existing} chunks). Use force=True to rebuild.")
        return existing

    pdfs = _policy_pdfs()
    if not pdfs:
        raise FileNotFoundError("No policy PDFs found under the data directory.")
    logger.info(f"Indexing {len(pdfs)} policy PDFs")

    nested = await asyncio.gather(*(_chunk_pdf(path) for path in pdfs))
    chunks = [chunk for group in nested for chunk in group]
    if not chunks:
        raise RuntimeError("Policy extraction produced no chunks.")
    logger.info(f"Extracted {len(chunks)} chunks; embedding with {cfg.embed_model}")

    embedder = Embedder(cfg.embed_model)
    vectors = embedder.embed([chunk.text for chunk in chunks])

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
    (index_dir / META_FILE).write_text(
        json.dumps(
            {"embed_model": cfg.embed_model, "chunk_count": len(chunks), "dim": int(vectors.shape[1])},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote policy index with {len(chunks)} chunks to {index_dir}")
    return len(chunks)
