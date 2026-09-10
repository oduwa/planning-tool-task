#!/usr/bin/env python3
"""Score the pipeline on the labelled training cases (001-005).

Runs the full pipeline for each training case, compares the predicted Decision
against the ground-truth ``decision.json`` using the provided
``PlanningDecisionSimpleEvaluator`` (decision match + semantic reasoning F1), and
prints per-case and aggregate scores. Use this to tune prompts and retrieval
before generating the holdout predictions.

Usage:
    uv run python scripts/evaluate_dev.py                 # all training cases
    uv run python scripts/evaluate_dev.py case-001 case-002
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from decision import Decision  # noqa: E402
from evaluator.evaluator import PlanningDecisionSimpleEvaluator  # noqa: E402
from llm import ModelConfig, get_client  # noqa: E402
from paths import DECISIONS_DIR, decision_path  # noqa: E402
from pipeline import run_case  # noqa: E402
from policy.retriever import PolicyRetriever  # noqa: E402

JUDGE_MODEL_ENV = "PLANNING_JUDGE_MODEL"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


def _training_cases() -> list[str]:
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


async def _evaluate(cases: list[str]) -> None:
    """Run and score the pipeline for the given training cases.

    Args:
        cases: Case identifiers to evaluate.
    """
    import os

    cfg = ModelConfig.from_env()
    retriever = PolicyRetriever(cfg)
    judge_model = os.getenv(JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL)
    evaluator = PlanningDecisionSimpleEvaluator(model=judge_model, client=get_client())

    for case_id in cases:
        expected = Decision.model_validate_json(
            decision_path(case_id).read_text(encoding="utf-8")
        )
        predicted = await run_case(case_id, cfg, retriever)
        _log_comparison(case_id, predicted, expected)
        scores = await evaluator.ascore_item(case_id, expected, predicted)
        print(  # noqa: T201
            f"{case_id}: decision_match={scores['decision_match']} "
            f"reasoning_f1={scores.get('f1', 0.0):.3f} "
            f"(P={scores.get('precision', 0.0):.3f} R={scores.get('recall', 0.0):.3f})"
        )

    aggregate = evaluator.aggregate()
    print("\n=== Aggregate ===")  # noqa: T201
    for key, value in aggregate.items():
        print(f"{key}: {value:.4f}")  # noqa: T201


def main() -> None:
    """Parse arguments and run the dev evaluation."""
    cases = sys.argv[1:] or _training_cases()
    asyncio.run(_evaluate(cases))


if __name__ == "__main__":
    main()
