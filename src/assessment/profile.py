"""Build a structured CaseProfile from a cleaned case pack.

A single text pass over the form, statements, consultation replies, and the
drawings' embedded text. The application drawings are vector CAD sheets with no
raster images, but their embedded text carries the load-bearing facts (gate and
opening dimensions, heights, materials, and title-block drawing references), so
those documents are included in the text assembled for extraction.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from assessment.schema import CaseProfile
from ingest.case_loader import CasePack, DocType
from llm import ModelConfig, complete_json

MAX_DOC_CHARS = 6000
MAX_TOTAL_CHARS = 40000

_PROFILE_SYSTEM = (
    "You are a UK planning officer's assistant extracting the salient facts of a "
    "householder/full planning application from its document pack. Be precise and "
    "only record facts present in the documents. The pack includes the application "
    "drawings as text: capture the key dimensions (heights, widths, storeys, "
    "separation distances), proposed materials, and any drawing numbers, revisions "
    "and 'received' dates from drawing title blocks. For consultee positions, "
    "capture each statutory or internal consultee (highways, environmental health, "
    "conservation, Yorkshire Water, lead local flood authority, etc.) and whether "
    "they object, have no objection, or have no objection subject to conditions."
)


def _assemble_text(pack: CasePack) -> str:
    """Concatenate the text documents of a pack under labelled headers.

    Drawing documents are included because their embedded text holds dimensions,
    materials, and title-block references a decision depends on.

    Args:
        pack: The cleaned case pack.
    """
    order = [
        DocType.APPLICATION_FORM,
        DocType.DESIGN_ACCESS_STATEMENT,
        DocType.SUPPORTING_STATEMENT,
        DocType.ELEVATIONS,
        DocType.SITE_PLAN,
        DocType.FLOOR_PLAN,
        DocType.HERITAGE_STATEMENT,
        DocType.FLOOD_RISK_ASSESSMENT,
        DocType.CONSULTATION_REPLY,
        DocType.OTHER,
    ]
    sections: list[str] = []
    total = 0
    for doc_type in order:
        for doc in pack.by_type(doc_type):
            text = doc.text.strip()
            if not text:
                continue
            snippet = text[:MAX_DOC_CHARS]
            block = f"===== {doc_type.value.upper()} :: {doc.filename} =====\n{snippet}"
            sections.append(block)
            total += len(block)
            if total >= MAX_TOTAL_CHARS:
                return "\n\n".join(sections)
    return "\n\n".join(sections)


async def build_case_profile(
    pack: CasePack, client: AsyncOpenAI, cfg: ModelConfig
) -> CaseProfile:
    """Extract a structured CaseProfile via a single text pass.

    Args:
        pack: The cleaned case pack.
        client: OpenAI-compatible async client.
        cfg: Model configuration.
    """
    text = _assemble_text(pack)
    return await complete_json(
        client,
        cfg.profile_model,
        CaseProfile,
        system=_PROFILE_SYSTEM,
        user=f"Extract the case profile from the following documents:\n\n{text}",
        temperature=cfg.temperature,
        seed=cfg.seed,
        extra_body=cfg.provider_extra_body(),
    )
