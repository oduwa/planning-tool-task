"""Deterministic (no-API) tests for the guardrail layer.

These cover the safeguards that most directly protect against the failure modes
flagged for this task: hallucinated citations, structural inconsistency, and the
domain-specific approval bias.
"""

from __future__ import annotations

from assessment.schema import (
    CaseAssessment,
    CaseProfile,
    ConsulteePosition,
    HarmLevel,
    PlanningTheme,
    ThemeAssessment,
)
from decision import Decision, DecisionType
from guardrails import (
    Severity,
    check_approval_bias,
    check_consistency,
    check_grounding,
)


class _StubRetriever:
    """A retriever double that only knows about a fixed set of citations."""

    def __init__(self, known: set[str]) -> None:
        self._known = {k.lower() for k in known}

    def verify_citation(self, reference: str) -> bool:
        return reference.lower() in self._known


def _theme(theme: PlanningTheme, harm: HarmLevel, cites: list[str]) -> ThemeAssessment:
    return ThemeAssessment(
        theme=theme, finding=f"{theme.value} finding.", harm=harm, cited_policies=cites
    )


def test_grounding_blocks_hallucinated_refusal_citation() -> None:
    """A refusal citing a policy absent from the corpus is a blocker."""
    assessment = CaseAssessment(
        themes=[_theme(PlanningTheme.AMENITY, HarmLevel.UNRESOLVED, ["Policy 999"])]
    )
    retriever = _StubRetriever(known={"Policy 44"})
    findings = check_grounding(assessment, retriever)  # type: ignore[arg-type]
    assert any(f.severity == Severity.BLOCKER for f in findings)


def test_grounding_passes_verified_citation() -> None:
    """A citation present in the corpus produces no grounding finding."""
    assessment = CaseAssessment(
        themes=[_theme(PlanningTheme.HIGHWAYS, HarmLevel.CONDITIONABLE, ["Policy 44"])]
    )
    retriever = _StubRetriever(known={"Policy 44"})
    assert check_grounding(assessment, retriever) == []  # type: ignore[arg-type]


def test_consistency_flags_refuse_with_conditions() -> None:
    """A refusal must not carry conditions."""
    decision = Decision(
        decision=DecisionType.REFUSE, reasons=["r"], conditions=["should not be here"]
    )
    assessment = CaseAssessment(
        themes=[_theme(PlanningTheme.FLOOD, HarmLevel.UNRESOLVED, ["Policy 1"])]
    )
    findings = check_consistency(decision, assessment)
    assert any(f.severity == Severity.BLOCKER for f in findings)


def test_consistency_flags_approve_with_unresolved_theme() -> None:
    """Approving while a theme is unresolved violates the decision invariant."""
    decision = Decision(
        decision=DecisionType.APPROVE, reasons=["r"], conditions=["c1", "c2"]
    )
    assessment = CaseAssessment(
        themes=[_theme(PlanningTheme.HERITAGE, HarmLevel.UNRESOLVED, ["Policy 1"])]
    )
    findings = check_consistency(decision, assessment)
    assert any("unresolved harm" in f.message for f in findings)


def test_approval_bias_blocks_unengaged_objection() -> None:
    """An approval that ignores a statutory objection is blocked for review."""
    profile = CaseProfile(
        proposal_description="Single dwelling",
        consultee_positions=[
            ConsulteePosition(body="Highways", stance="objection", summary="Unsafe access")
        ],
    )
    assessment = CaseAssessment(
        themes=[_theme(PlanningTheme.PRINCIPLE, HarmLevel.NONE, ["Policy 1"])]
    )
    decision = Decision(decision=DecisionType.APPROVE, reasons=["r"], conditions=["c1", "c2"])
    findings = check_approval_bias(profile, assessment, decision)
    assert any(f.severity == Severity.BLOCKER for f in findings)


def test_approval_bias_ignores_no_objection() -> None:
    """A 'no objection' consultee does not trigger the approval-bias guardrail."""
    profile = CaseProfile(
        proposal_description="Single dwelling",
        consultee_positions=[
            ConsulteePosition(body="Highways", stance="no objection", summary="Fine")
        ],
    )
    assessment = CaseAssessment(
        themes=[_theme(PlanningTheme.HIGHWAYS, HarmLevel.NONE, ["Policy 44"])]
    )
    decision = Decision(decision=DecisionType.APPROVE, reasons=["r"], conditions=["c1", "c2"])
    blockers = [
        f for f in check_approval_bias(profile, assessment, decision)
        if f.severity == Severity.BLOCKER
    ]
    assert blockers == []
