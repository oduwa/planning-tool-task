#!/usr/bin/env python3
"""Evaluation-driven development harness for the planning decision tool.

Runs the full end-to-end pipeline for each labelled case, scores the predicted
Decision against the ground-truth ``decision.json`` (decision match + semantic
reasoning F1), and — crucially — writes the evidence to a committed
``evaluation/runs/<timestamp>/`` directory so it reaches the submission:

- ``metrics.json``   — end-to-end task metrics (decision accuracy is the headline,
  not just component F1), a confusion matrix, and the model config used.
- ``cases/<id>.json``— a per-case trace: predicted vs expected decision, scores,
  the profile, the per-theme assessment, and the guardrail report.
- ``error_analysis.md`` — a human-readable diff of every case, with mismatches and
  guardrail flags called out, to drive the next iteration.

The headline number is the **task outcome** (did we get approve/refuse right),
because strong component metrics can still hide poor end-to-end performance.

Usage:
    uv run python scripts/evaluate_dev.py                 # all labelled cases
    uv run python scripts/evaluate_dev.py case-001 case-002
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from decision import Decision  # noqa: E402
from evaluator.evaluator import PlanningDecisionSimpleEvaluator  # noqa: E402
from llm import ModelConfig, get_client  # noqa: E402
from paths import DECISIONS_DIR, REPO_ROOT, decision_path  # noqa: E402
from pipeline import CaseResult, run_case_detailed  # noqa: E402
from policy.retriever import PolicyRetriever  # noqa: E402

JUDGE_MODEL_ENV = "PLANNING_JUDGE_MODEL"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
EVAL_DIR = REPO_ROOT / "evaluation"


def _labelled_cases() -> list[str]:
    """Return the case ids that have a ground-truth decision."""
    return sorted(path.name for path in DECISIONS_DIR.iterdir() if path.is_dir())


def _format_decision(decision: Decision) -> str:
    """Render a decision's outcome, reasons, and conditions as an indented block.

    Args:
        decision: The decision to render.
    """
    lines = [f"  outcome: {decision.decision.value}"]
    lines.append(f"  reasons ({len(decision.reasons)}):")
    lines.extend(f"    - {reason}" for reason in decision.reasons)
    lines.append(f"  conditions ({len(decision.conditions)}):")
    lines.extend(f"    - {condition}" for condition in decision.conditions)
    return "\n".join(lines)


def _log_comparison(case_id: str, predicted: Decision, expected: Decision) -> None:
    """Print the model output next to the ground truth for a case.

    Args:
        case_id: Case identifier.
        predicted: The pipeline's predicted decision.
        expected: The ground-truth decision.
    """
    print(f"\n{'=' * 70}")  # noqa: T201
    print(f"{case_id}: model output vs. ground truth")  # noqa: T201
    print(f"{'-' * 70}")  # noqa: T201
    print("[PREDICTED]")  # noqa: T201
    print(_format_decision(predicted))  # noqa: T201
    print("[GROUND TRUTH]")  # noqa: T201
    print(_format_decision(expected))  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201


def _case_trace(
    result: CaseResult, expected: Decision | None, scores: dict[str, float]
) -> dict[str, object]:
    """Build the per-case JSON trace written to the run directory.

    Args:
        result: The detailed pipeline result.
        expected: The ground-truth decision, if available.
        scores: The evaluator scores for this case.
    """
    return {
        "case_id": result.case_id,
        "predicted": result.decision.model_dump(),
        "expected": expected.model_dump() if expected else None,
        "scores": scores,
        "guardrails": {
            "requires_human_review": result.guardrails.requires_human_review,
            "findings": [f.model_dump() for f in result.guardrails.findings],
        },
        "profile": result.profile.model_dump(),
        "assessment": result.assessment.model_dump(),
        "dropped_documents": result.dropped_documents,
    }


def _write_error_analysis(
    out_dir: Path, records: list[dict[str, object]]
) -> None:
    """Write a human-readable error-analysis report for the run.

    Args:
        out_dir: The run output directory.
        records: The per-case trace records.
    """
    lines = ["# Error analysis", ""]
    for record in records:
        case_id = record["case_id"]
        predicted = Decision.model_validate(record["predicted"])
        expected_raw = record["expected"]
        scores = record["scores"]  # type: ignore[assignment]
        guardrails = record["guardrails"]  # type: ignore[assignment]
        lines.append(f"## {case_id}")
        if expected_raw is None:
            lines.append("- No ground truth (holdout case); trace only.")
        else:
            expected = Decision.model_validate(expected_raw)
            match = expected.decision == predicted.decision
            lines.append(
                f"- Decision: predicted **{predicted.decision.value}**, "
                f"expected **{expected.decision.value}** "
                f"({'MATCH' if match else 'MISMATCH'})."
            )
            lines.append(
                f"- Reasoning F1: {scores.get('f1', 0.0):.3f} "  # type: ignore[union-attr]
                f"(P={scores.get('precision', 0.0):.3f} "  # type: ignore[union-attr]
                f"R={scores.get('recall', 0.0):.3f})."  # type: ignore[union-attr]
            )
            if not match:
                lines.append("- **Divergence** — inspect reasons below:")
                lines.append(f"  - predicted reasons: {predicted.reasons}")
                lines.append(f"  - expected reasons: {expected.reasons}")
        review = guardrails["requires_human_review"]  # type: ignore[index]
        findings = guardrails["findings"]  # type: ignore[index]
        lines.append(f"- Guardrails: human_review={review}, {len(findings)} finding(s).")
        for finding in findings:  # type: ignore[union-attr]
            lines.append(
                f"  - [{finding['severity']}] {finding['check']}: {finding['message']}"
            )
        lines.append("")
    (out_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


async def _evaluate(cases: list[str]) -> None:
    """Run and score the pipeline for the given cases, writing committed artifacts.

    Args:
        cases: Case identifiers to evaluate.
    """
    cfg = ModelConfig.from_env()
    retriever = PolicyRetriever(cfg)
    judge_model = os.getenv(JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL)
    evaluator = PlanningDecisionSimpleEvaluator(model=judge_model, client=get_client())

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = EVAL_DIR / "runs" / timestamp
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    confusion: Counter[str] = Counter()
    decision_hits = 0
    scored = 0

    for case_id in cases:
        expected: Decision | None = None
        if decision_path(case_id).is_file():
            expected = Decision.model_validate_json(
                decision_path(case_id).read_text(encoding="utf-8")
            )

        result = await run_case_detailed(case_id, cfg, retriever)
        scores: dict[str, float] = {}
        if expected is not None:
            _log_comparison(case_id, result.decision, expected)
            scores = await evaluator.ascore_item(case_id, expected, result.decision)
            scored += 1
            decision_hits += int(scores.get("decision_match", 0.0))
            confusion[f"{expected.decision.value}->{result.decision.decision.value}"] += 1
            review = "REVIEW" if result.guardrails.requires_human_review else "ok"
            print(  # noqa: T201
                f"{case_id}: decision_match={scores['decision_match']} "
                f"reasoning_f1={scores.get('f1', 0.0):.3f} "
                f"(P={scores.get('precision', 0.0):.3f} R={scores.get('recall', 0.0):.3f}) "
                f"guardrails={review}"
            )

        record = _case_trace(result, expected, scores)
        records.append(record)
        (out_dir / "cases" / f"{case_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    aggregate = evaluator.aggregate() if scored else {}
    metrics = {
        "timestamp": timestamp,
        "cases": cases,
        "scored_cases": scored,
        "decision_accuracy": (decision_hits / scored) if scored else None,
        "confusion_matrix": dict(confusion),
        "reasoning": aggregate,
        "model_config": {
            "profile_model": cfg.profile_model,
            "reasoning_model": cfg.reasoning_model,
            "vision_model": cfg.vision_model,
            "embed_model": cfg.embed_model,
            "enable_vision": cfg.enable_vision,
            "seed": cfg.seed,
            "judge_model": judge_model,
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_error_analysis(out_dir, records)

    print("\n=== Aggregate ===")  # noqa: T201
    if scored:
        print(f"decision_accuracy: {metrics['decision_accuracy']:.4f} "  # noqa: T201
              f"over {scored} case(s)")
        print(f"confusion (expected->predicted): {dict(confusion)}")  # noqa: T201
    for key, value in aggregate.items():
        print(f"{key}: {value:.4f}")  # noqa: T201
    print(f"\nArtifacts written to {out_dir}")  # noqa: T201


def main() -> None:
    """Parse arguments and run the evaluation."""
    cases = sys.argv[1:] or _labelled_cases()
    asyncio.run(_evaluate(cases))


if __name__ == "__main__":
    main()
