"""End-to-end orchestration: case folder in, Decision out.

Wires the stages together: ingest -> profile (text + vision) -> assess (agent
with policy retrieval + constraints) -> synthesize -> Decision. The policy
retriever is loaded once and can be reused across many cases in a batch run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from assessment.agent import assess_case
from assessment.profile import build_case_profile
from assessment.synthesize import synthesize_decision
from decision import Decision
from ingest.case_loader import load_case_pack
from llm import ModelConfig, get_client
from paths import DRAFTS_DIR, PREDICTIONS_DIR, application_dir, prediction_path
from policy.retriever import PolicyRetriever


async def run_case(
    case_id: str,
    cfg: ModelConfig,
    retriever: PolicyRetriever,
) -> Decision:
    """Run the full pipeline for one case and return its Decision.

    Args:
        case_id: Case identifier such as ``case-006``.
        cfg: Model configuration.
        retriever: Loaded policy retriever, reused across cases.
    """
    client = get_client()
    logger.info(f"[{case_id}] loading and cleaning pack")
    pack = await load_case_pack(application_dir(case_id))

    logger.info(f"[{case_id}] building case profile")
    profile = await build_case_profile(pack, client, cfg)

    logger.info(f"[{case_id}] assessing themes")
    assessment = await assess_case(profile, client, cfg, retriever)

    decision = synthesize_decision(assessment, profile)
    logger.info(
        f"[{case_id}] decision={decision.decision.value} "
        f"reasons={len(decision.reasons)} conditions={len(decision.conditions)}"
    )
    return decision


def write_outputs(case_id: str, decision: Decision) -> Path:
    """Write the prediction JSON and an officer-facing markdown draft.

    Args:
        case_id: Case identifier such as ``case-006``.
        decision: The synthesized decision.

    Returns:
        The path to the written prediction JSON.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    output = prediction_path(case_id)
    payload = {
        "decision": decision.decision.value,
        "reasons": decision.reasons,
        "conditions": decision.conditions,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    decision.to_markdown_file(
        DRAFTS_DIR / f"{case_id}-draft.md",
        batch_name=case_id,
        local_authority="Doncaster Council",
        created_at=datetime.now(UTC),
    )
    return output
