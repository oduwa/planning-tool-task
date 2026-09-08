"""Build a structured CaseProfile from a cleaned case pack.

Two passes: a text pass over the form, statements, and consultation replies, and
a vision pass over the drawings to recover dimensions and drawing-title-block
references the text usually omits. The two are merged into one profile.
"""

from __future__ import annotations

from loguru import logger
from openai import AsyncOpenAI

from assessment.schema import CaseProfile, PlanReadout
from ingest.case_loader import CasePack, DocType
from llm import ModelConfig, complete_json, image_data_uri

MAX_DOC_CHARS = 6000
MAX_TOTAL_CHARS = 40000

_PROFILE_SYSTEM = (
    "You are a UK planning officer's assistant extracting the salient facts of a "
    "householder/full planning application from its document pack. Be precise and "
    "only record facts present in the documents. For consultee positions, capture "
    "each statutory or internal consultee (highways, environmental health, "
    "conservation, Yorkshire Water, lead local flood authority, etc.) and whether "
    "they object, have no objection, or have no objection subject to conditions."
)

_VISION_SYSTEM = (
    "You are a UK planning officer reading application drawings (elevations, site "
    "plans, floor plans). Extract measurable facts a decision depends on: ridge and "
    "eaves heights, number of storeys, separation distances to boundaries and "
    "neighbours, plot coverage, parking provision, and any drawing numbers, "
    "revisions and 'received' dates visible in title blocks."
)


def _assemble_text(pack: CasePack) -> str:
    """Concatenate the text documents of a pack under labelled headers.

    Args:
        pack: The cleaned case pack.
    """
    order = [
        DocType.APPLICATION_FORM,
        DocType.DESIGN_ACCESS_STATEMENT,
        DocType.SUPPORTING_STATEMENT,
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


def _plan_image_uris(pack: CasePack, limit: int) -> list[str]:
    """Collect base64 image URIs from plan documents, capped at ``limit``.

    Args:
        pack: The cleaned case pack.
        limit: Maximum number of images to include.
    """
    uris: list[str] = []
    for doc in pack.plans:
        for block in doc.document.image_blocks:
            uris.append(image_data_uri(block.image_bytes, block.image_format))
            if len(uris) >= limit:
                return uris
    return uris


async def build_case_profile(
    pack: CasePack, client: AsyncOpenAI, cfg: ModelConfig
) -> CaseProfile:
    """Extract a structured CaseProfile via a text pass plus a vision pass.

    Args:
        pack: The cleaned case pack.
        client: OpenAI-compatible async client.
        cfg: Model configuration.
    """
    text = _assemble_text(pack)
    profile = await complete_json(
        client,
        cfg.chat_model,
        CaseProfile,
        system=_PROFILE_SYSTEM,
        user=f"Extract the case profile from the following documents:\n\n{text}",
        temperature=cfg.temperature,
    )

    image_uris = _plan_image_uris(pack, cfg.max_vision_images)
    if image_uris:
        try:
            readout = await complete_json(
                client,
                cfg.vision_model,
                PlanReadout,
                system=_VISION_SYSTEM,
                user=(
                    "Read these application drawings and report the measurable facts "
                    "and any drawing title-block references."
                ),
                image_uris=image_uris,
                temperature=cfg.temperature,
            )
            profile = _merge_readout(profile, readout)
        except (ValueError, Exception) as error:  # noqa: BLE001 - vision is best-effort
            logger.warning(f"Vision pass failed for {pack.case_id}: {error}")
    else:
        logger.info(f"No plan images available for {pack.case_id}; skipping vision pass")

    return profile


def _merge_readout(profile: CaseProfile, readout: PlanReadout) -> CaseProfile:
    """Fold vision-pass findings into the text profile without losing facts.

    Args:
        profile: The text-derived profile.
        readout: The vision-derived readout.
    """
    dimensions = list(dict.fromkeys([*profile.key_dimensions, *readout.key_dimensions]))

    existing = {d.as_clause() for d in profile.drawing_references}
    drawings = list(profile.drawing_references)
    for ref in readout.drawing_references:
        if ref.as_clause() not in existing:
            drawings.append(ref)

    notes = profile.raw_notes
    if readout.layout_notes:
        notes = f"{notes}\nPlan reading: {readout.layout_notes}".strip()

    return profile.model_copy(
        update={
            "key_dimensions": dimensions,
            "drawing_references": drawings,
            "raw_notes": notes,
        }
    )
