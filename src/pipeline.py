"""End-to-end orchestration: case folder in, Decision (+ audit trail) out.

Stage order: ingest -> profile (text) -> plan vision pass (rasterised drawings,
merged into the profile) -> iterative policy assessment -> deterministic
synthesis -> guardrails. The policy retriever is loaded once and reused across a
batch. ``run_case`` returns just the ``Decision`` for existing callers;
``run_case_detailed`` returns the full ``CaseResult`` (profile, assessment,
guardrail report) used for officer drafts and evaluation traces.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from assessment.agent import assess_case
from assessment.profile import build_case_profile
from assessment.schema import CaseAssessment, CaseProfile
from assessment.synthesize import synthesize_decision
from assessment.vision import merge_readout, read_plans
from decision import Decision
from guardrails import GuardrailReport, run_guardrails
from ingest.case_loader import load_case_pack
from llm import ModelConfig, get_client
from paths import DRAFTS_DIR, PREDICTIONS_DIR, application_dir, prediction_path
from policy.retriever import PolicyRetriever


class CaseResult(BaseModel):
    """The full result of a pipeline run for one case, for drafts and eval traces."""

    case_id: str
    decision: Decision
    profile: CaseProfile
    assessment: CaseAssessment
    guardrails: GuardrailReport
    dropped_documents: list[str] = []


async def run_case_detailed(
    case_id: str,
    cfg: ModelConfig,
    retriever: PolicyRetriever,
) -> CaseResult:
    """Run the full pipeline for one case and return the detailed result.

    Args:
        case_id: Case identifier such as ``case-006``.
        cfg: Model configuration.
        retriever: Loaded policy retriever, reused across cases.
    """
    client = get_client()
    logger.info(f"[{case_id}] loading and cleaning pack")
    pack = await load_case_pack(application_dir(case_id))

    logger.info(f"[{case_id}] building case profile (text)")
    profile = await build_case_profile(pack, client, cfg)

    readout = await read_plans(pack, client, cfg)
    profile = merge_readout(profile, readout)

    logger.info(f"[{case_id}] assessing themes")
    assessment = await assess_case(profile, client, cfg, retriever)

    decision = synthesize_decision(assessment, profile)
    guardrails = run_guardrails(profile, assessment, decision, retriever)
    review = "REVIEW REQUIRED" if guardrails.requires_human_review else "clear"
    logger.info(
        f"[{case_id}] decision={decision.decision.value} "
        f"reasons={len(decision.reasons)} conditions={len(decision.conditions)} "
        f"guardrails={review} ({len(guardrails.blockers)} blocker(s), "
        f"{len(guardrails.warnings)} warning(s))"
    )
    return CaseResult(
        case_id=case_id,
        decision=decision,
        profile=profile,
        assessment=assessment,
        guardrails=guardrails,
        dropped_documents=pack.dropped,
    )


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
    result = await run_case_detailed(case_id, cfg, retriever)
    return result.decision


def write_outputs(case_id: str, result: CaseResult) -> Path:
    """Write the prediction JSON and an officer-facing markdown draft.

    The draft embeds the guardrail report so an officer sees, alongside the draft
    decision, whether automated checks flagged it for review.

    Args:
        case_id: Case identifier such as ``case-006``.
        result: The detailed pipeline result.

    Returns:
        The path to the written prediction JSON.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    decision = result.decision
    output = prediction_path(case_id)
    payload = {
        "decision": decision.decision.value,
        "reasons": decision.reasons,
        "conditions": decision.conditions,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    draft = decision.to_markdown(
        batch_name=case_id,
        local_authority="Doncaster Council",
        created_at=datetime.now(UTC),
    )
    draft = f"{draft}\n\n{result.guardrails.to_markdown()}"
    (DRAFTS_DIR / f"{case_id}-draft.md").write_text(draft, encoding="utf-8")
    return output
