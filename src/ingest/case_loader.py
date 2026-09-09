"""Load a messy application pack into a clean, ordered set of documents.

The packs deliberately contain traps: every file appears twice (hyphenated and
underscored) and some drawings/statements ship superseded or refused variants.
This module collapses duplicates, drops withdrawn versions, and classifies what
remains so downstream stages know which documents are plans (for the vision
pass) and which are consultee replies (which carry objections).
"""

from __future__ import annotations

import asyncio
import re
from enum import StrEnum
from pathlib import Path

from kreuzberg import (  # type: ignore[attr-defined]
    ExtractionConfig,
    ImageExtractionConfig,
    PageConfig,
    PdfConfig,
    ResultFormat,
)
from loguru import logger
from pydantic import BaseModel, Field

from parser.document import PdfDocument
from parser.parser import extract_pdf_document
from paths import extracted_json_path

SUPERSEDED_MARKERS = ("superseded", "refused", "withdrawn", "old", "previous")


class DocType(StrEnum):
    """Coarse classification of a case document."""

    APPLICATION_FORM = "application_form"
    DESIGN_ACCESS_STATEMENT = "design_access_statement"
    HERITAGE_STATEMENT = "heritage_statement"
    FLOOD_RISK_ASSESSMENT = "flood_risk_assessment"
    ELEVATIONS = "elevations"
    SITE_PLAN = "site_plan"
    FLOOR_PLAN = "floor_plan"
    CONSULTATION_REPLY = "consultation_reply"
    SUPPORTING_STATEMENT = "supporting_statement"
    OTHER = "other"


PLAN_TYPES = {DocType.ELEVATIONS, DocType.SITE_PLAN, DocType.FLOOR_PLAN}


class CaseDocument(BaseModel):
    """A single parsed, classified document from a case pack."""

    doc_type: DocType
    filename: str
    document: PdfDocument
    is_plan: bool = Field(
        default=False, description="Whether this is a drawing suited to a vision pass"
    )

    @property
    def text(self) -> str:
        """Return the concatenated text of the document."""
        return self.document.text


class CasePack(BaseModel):
    """The cleaned set of documents for one planning application."""

    case_id: str
    documents: list[CaseDocument] = Field(default_factory=list)
    dropped: list[str] = Field(
        default_factory=list,
        description="Files skipped as duplicates or superseded, for the audit trail",
    )

    def by_type(self, *types: DocType) -> list[CaseDocument]:
        """Return documents whose type is in ``types``.

        Args:
            types: One or more document types to filter by.
        """
        wanted = set(types)
        return [doc for doc in self.documents if doc.doc_type in wanted]

    @property
    def plans(self) -> list[CaseDocument]:
        """Return documents that are drawings (elevations, site, floor plans)."""
        return [doc for doc in self.documents if doc.is_plan]


def _text_only_config() -> ExtractionConfig:
    """Build a lightweight text-only extraction config for non-plan documents."""
    return ExtractionConfig(
        result_format=ResultFormat.ELEMENT_BASED,
        pages=PageConfig(extract_pages=True),
        images=ImageExtractionConfig(extract_images=False),
        pdf_options=PdfConfig(extract_metadata=True),
    )


def _normalized_stem(filename: str) -> str:
    """Return a stem with hyphen/underscore runs collapsed, for dedupe keys.

    Args:
        filename: Original file name.
    """
    stem = Path(filename).stem.lower()
    return re.sub(r"[-_]+", "_", stem).strip("_")


def _is_superseded(filename: str) -> bool:
    """Return True if the filename signals a superseded/refused/withdrawn version.

    Args:
        filename: Original file name.
    """
    stem = _normalized_stem(filename)
    return any(marker in stem for marker in SUPERSEDED_MARKERS)


def _base_key(filename: str) -> str:
    """Return the dedupe key: normalized stem minus any superseded marker.

    So ``existing_and_proposed_elevations`` and
    ``existing-and-proposed-elevations-superseded`` share a key and compete.

    Args:
        filename: Original file name.
    """
    stem = _normalized_stem(filename)
    for marker in SUPERSEDED_MARKERS:
        stem = stem.replace(f"_{marker}", "").replace(marker, "")
    return stem.strip("_")


def classify_document(filename: str, text: str) -> DocType:
    """Classify a document from its filename, falling back to text signals.

    Args:
        filename: Original file name.
        text: Extracted document text (first portion is enough).
    """
    stem = _normalized_stem(filename)
    head = text[:2000].lower()

    if "application_form" in stem or "application form" in head:
        return DocType.APPLICATION_FORM
    if "heritage" in stem or "heritage statement" in head:
        return DocType.HERITAGE_STATEMENT
    if "flood" in stem or "flood risk" in head:
        return DocType.FLOOD_RISK_ASSESSMENT
    if "design_and_access" in stem or "design and access" in head:
        return DocType.DESIGN_ACCESS_STATEMENT
    if "elevation" in stem:
        return DocType.ELEVATIONS
    if "site_plan" in stem or ("site" in stem and "plan" in stem):
        return DocType.SITE_PLAN
    if "floor_plan" in stem or ("floor" in stem and "plan" in stem):
        return DocType.FLOOR_PLAN
    if "consultation" in stem or "consultation reply" in head or "consultee" in head:
        return DocType.CONSULTATION_REPLY
    if "statement" in stem or "supporting" in stem:
        return DocType.SUPPORTING_STATEMENT
    return DocType.OTHER


def _cached_json_path(path: Path) -> Path | None:
    """Return the ``uv run parser`` cache path for a PDF if it exists.

    Args:
        path: Path to the source PDF.
    """
    try:
        candidate = extracted_json_path(path)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


async def _parse(path: Path, *, plan: bool, use_cache: bool) -> PdfDocument:
    """Load a PDF from the parser cache, else parse it live.

    Cached documents come from ``uv run parser`` (default config, so they always
    include images); the vision stage simply ignores images on non-plan docs.

    Args:
        path: Path to the PDF.
        plan: Whether this is a drawing that needs image extraction.
        use_cache: Whether to prefer pre-extracted JSON under ``data/extracted``.
    """
    if use_cache:
        cache_path = _cached_json_path(path)
        if cache_path is not None:
            try:
                return PdfDocument.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
            except (ValueError, OSError) as error:
                logger.warning(
                    f"Ignoring unreadable parser cache {cache_path.name}: {error}"
                )
    config = None if plan else _text_only_config()
    return await extract_pdf_document(path, config=config)


def _select_survivors(case_dir: Path) -> tuple[list[Path], list[str]]:
    """Pick one file per dedupe key, preferring live over superseded versions.

    Args:
        case_dir: Directory holding the raw case PDFs.

    Returns:
        A tuple of (surviving paths, dropped file names).
    """
    pdfs = sorted(case_dir.glob("*.pdf"))
    groups: dict[str, list[Path]] = {}
    for path in pdfs:
        groups.setdefault(_base_key(path.name), []).append(path)

    survivors: list[Path] = []
    dropped: list[str] = []
    for _key, members in groups.items():
        live = [p for p in members if not _is_superseded(p.name)]
        candidates = live or members
        # Prefer underscore filenames deterministically, then largest on disk.
        chosen = max(candidates, key=lambda p: (p.stat().st_size, "_" in p.name))
        survivors.append(chosen)
        dropped.extend(p.name for p in members if p != chosen)

    return sorted(survivors), sorted(dropped)


async def load_case_pack(case_dir: Path, *, use_cache: bool = True) -> CasePack:
    """Load, de-duplicate, classify, and parse a case application pack.

    Args:
        case_dir: Directory containing the raw application PDFs.
        use_cache: Prefer pre-extracted JSON from ``uv run parser`` when present,
            falling back to live parsing for any file missing from the cache.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    survivors, dropped = _select_survivors(case_dir)
    cached = sum(1 for p in survivors if use_cache and _cached_json_path(p) is not None)
    logger.info(
        f"{case_dir.name}: {len(survivors)} documents kept, {len(dropped)} dropped, "
        f"{cached} loaded from parser cache"
    )

    is_plan_flags = [
        classify_document(path.name, "") in PLAN_TYPES for path in survivors
    ]
    parsed = await asyncio.gather(
        *(
            _parse(path, plan=plan, use_cache=use_cache)
            for path, plan in zip(survivors, is_plan_flags, strict=True)
        )
    )

    documents: list[CaseDocument] = []
    for path, document in zip(survivors, parsed, strict=True):
        doc_type = classify_document(path.name, document.text)
        documents.append(
            CaseDocument(
                doc_type=doc_type,
                filename=path.name,
                document=document,
                is_plan=doc_type in PLAN_TYPES,
            )
        )

    return CasePack(case_id=case_dir.name, documents=documents, dropped=dropped)
