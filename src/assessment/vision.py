"""Multimodal plan reading: rasterise vector CAD drawings and read them with vision.

The README requires officers to "reason across all types of provided policy
materials including policy guidance, diagrams & measurements". The application
drawings are vector CAD sheets with *no embedded raster images*, so the PDF
parser's image extraction returns nothing for them — a text-only pipeline is
blind to the geometry. We therefore rasterise each drawing page with PyMuPDF and
send the resulting PNGs to a vision model, which recovers ridge/eaves heights,
storeys, separation distances, parking and title-block drawing references that
the flattened CAD text frequently garbles or drops.

This stage is optional and controlled by ``ModelConfig.enable_vision`` so the
pipeline can fall back to text-only (e.g. to save cost) without code changes.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from assessment.schema import CaseProfile, DrawingRef, PlanReadout
from ingest.case_loader import CasePack
from llm import ModelConfig, complete_json, image_data_uri

# 2x zoom over the 72 dpi PDF base gives ~144 dpi, enough for dimension text
# without producing needlessly large images.
RENDER_ZOOM = 2.0

_VISION_SYSTEM = (
    "You are a UK planning officer reading the submitted drawings for a housing "
    "application. You are shown rasterised CAD sheets (elevations, site plans, "
    "floor plans). Read them like an officer: extract the numeric dimensions that "
    "a decision depends on — ridge and eaves heights, number of storeys, overall "
    "widths and depths, separation distances to boundaries and neighbours, gate or "
    "opening widths, and parking provision. Record materials annotated on the "
    "drawings. Read each title block for the drawing title, drawing number, "
    "revision and any 'received'/date stamp. Only record what is actually legible "
    "on the sheets; never invent a dimension. If a value is unclear, omit it."
)


def _render_plan_images(pack: CasePack, max_pages: int) -> list[str]:
    """Rasterise the pack's drawing pages to base64 PNG data URIs.

    Args:
        pack: The cleaned case pack.
        max_pages: Total page budget across all drawings (cost cap).

    Returns:
        A list of ``data:image/png;base64,...`` URIs in document order.

    Raises:
        RuntimeError: If PyMuPDF is not installed but rasterisation is needed.
    """
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyMuPDF is required to rasterise plan drawings for the vision pass. "
            "Install project dependencies with `uv sync`, or disable vision by "
            "setting PLANNING_ENABLE_VISION=0."
        ) from error

    uris: list[str] = []
    for doc in pack.plans:
        if len(uris) >= max_pages:
            break
        path = Path(doc.source_path)
        if not path.is_file():
            logger.warning(f"Plan source missing, cannot rasterise: {path}")
            continue
        try:
            with pymupdf.open(path) as pdf:
                for page in pdf:
                    if len(uris) >= max_pages:
                        break
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM)
                    )
                    uris.append(image_data_uri(pixmap.tobytes("png"), "png"))
        except Exception as error:  # noqa: BLE001 - keep the pipeline resilient
            logger.warning(f"Failed to rasterise {path.name}: {error}")
    return uris


async def read_plans(
    pack: CasePack, client: AsyncOpenAI, cfg: ModelConfig
) -> PlanReadout | None:
    """Run the vision pass over the pack's drawings.

    Args:
        pack: The cleaned case pack.
        client: OpenAI-compatible async client.
        cfg: Model configuration (supplies the vision model and page budget).

    Returns:
        A ``PlanReadout`` when drawings were read, else ``None`` (no drawings, or
        vision disabled by config).
    """
    if not cfg.enable_vision or not pack.plans:
        return None

    image_uris = _render_plan_images(pack, cfg.max_vision_pages)
    if not image_uris:
        return None

    plan_names = ", ".join(doc.filename for doc in pack.plans)
    logger.info(
        f"{pack.case_id}: vision pass over {len(image_uris)} rendered page(s) "
        f"from drawings [{plan_names}] with {cfg.vision_model}"
    )
    return await complete_json(
        client,
        cfg.vision_model,
        PlanReadout,
        system=_VISION_SYSTEM,
        user=(
            "Read the attached drawing sheets and extract the dimensions, materials "
            "and title-block references as structured JSON. The sheets are: "
            f"{plan_names}."
        ),
        image_uris=image_uris,
        temperature=cfg.temperature,
        seed=cfg.seed,
        extra_body=cfg.provider_extra_body(),
    )


def _drawing_key(ref: DrawingRef) -> str:
    """Return a dedupe key for a drawing reference.

    Args:
        ref: The drawing reference.
    """
    return (ref.number or ref.description).strip().lower()


def merge_readout(profile: CaseProfile, readout: PlanReadout | None) -> CaseProfile:
    """Fold vision-recovered facts into the text-extracted profile.

    Vision augments rather than overrides: text-extracted dimensions and drawing
    references are kept, and only genuinely new ones from the drawings are added.
    Layout observations are appended to the profile notes with a clear provenance
    marker so the audit trail shows what came from the drawings.

    Args:
        profile: The profile built from document text.
        readout: The vision readout, or ``None`` when vision did not run.
    """
    if readout is None:
        return profile

    existing_dims = {d.strip().lower() for d in profile.key_dimensions}
    merged_dims = list(profile.key_dimensions)
    for dim in readout.key_dimensions:
        if dim.strip().lower() not in existing_dims:
            merged_dims.append(dim)
            existing_dims.add(dim.strip().lower())

    existing_refs = {_drawing_key(r) for r in profile.drawing_references}
    merged_refs = list(profile.drawing_references)
    for ref in readout.drawing_references:
        if _drawing_key(ref) not in existing_refs:
            merged_refs.append(ref)
            existing_refs.add(_drawing_key(ref))

    notes = profile.raw_notes
    if readout.layout_notes.strip():
        notes = (
            f"{notes}\n\n[From drawings] {readout.layout_notes.strip()}"
            if notes
            else f"[From drawings] {readout.layout_notes.strip()}"
        )

    return profile.model_copy(
        update={
            "key_dimensions": merged_dims,
            "drawing_references": merged_refs,
            "raw_notes": notes,
        }
    )
