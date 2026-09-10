"""Deterministic (no-API) tests for synthesis, citation matching, and plan merge."""

from __future__ import annotations

from assessment.schema import (
    CaseAssessment,
    CaseProfile,
    DrawingRef,
    HarmLevel,
    PlanningTheme,
    PlanReadout,
    ThemeAssessment,
)
from assessment.synthesize import synthesize_decision
from assessment.vision import merge_readout
from decision import DecisionType
from policy.retriever import PolicyRetriever


def _theme(theme: PlanningTheme, harm: HarmLevel, condition: str | None = None) -> ThemeAssessment:
    return ThemeAssessment(
        theme=theme,
        finding=f"{theme.value} finding.",
        harm=harm,
        cited_policies=["Policy 1"],
        suggested_condition=condition,
    )


def test_synthesis_refuses_on_any_unresolved() -> None:
    """Any unresolved theme forces refusal with no conditions."""
    assessment = CaseAssessment(
        themes=[
            _theme(PlanningTheme.PRINCIPLE, HarmLevel.NONE),
            _theme(PlanningTheme.AMENITY, HarmLevel.UNRESOLVED),
        ]
    )
    decision = synthesize_decision(assessment, CaseProfile(proposal_description="x"))
    assert decision.decision == DecisionType.REFUSE
    assert decision.conditions == []


def test_synthesis_approves_with_standard_conditions() -> None:
    """With no unresolved harm, approval carries the two standard conditions."""
    assessment = CaseAssessment(
        themes=[
            _theme(PlanningTheme.PRINCIPLE, HarmLevel.NONE),
            _theme(PlanningTheme.HIGHWAYS, HarmLevel.CONDITIONABLE, "Provide parking. Reason: Policy 44."),
        ]
    )
    decision = synthesize_decision(assessment, CaseProfile(proposal_description="x"))
    assert decision.decision == DecisionType.APPROVE
    # commencement + approved-plans + the theme condition
    assert len(decision.conditions) == 3


def test_reference_pattern_matches_variants() -> None:
    """The citation matcher tolerates spacing and para/paragraph variants."""
    policy = PolicyRetriever._reference_pattern("Policy 44")
    assert policy is not None and policy.search("contrary to policy  44 of the plan")
    para = PolicyRetriever._reference_pattern("NPPF paragraph 135")
    assert para is not None and para.search("see para 135 here")
    assert PolicyRetriever._reference_pattern("the design guide") is None


def test_merge_readout_adds_new_dimensions_and_notes() -> None:
    """Vision readout augments the profile without dropping existing facts."""
    profile = CaseProfile(
        proposal_description="Extension",
        key_dimensions=["Eaves height 2.4m"],
        drawing_references=[DrawingRef(description="Proposed Elevations")],
    )
    readout = PlanReadout(
        key_dimensions=["Eaves height 2.4m", "Ridge height 5.1m"],
        drawing_references=[DrawingRef(description="Proposed Site Plan", number="102")],
        layout_notes="Parking to the front.",
    )
    merged = merge_readout(profile, readout)
    assert "Ridge height 5.1m" in merged.key_dimensions
    assert len(merged.key_dimensions) == 2  # no duplicate of the eaves height
    assert any(r.number == "102" for r in merged.drawing_references)
    assert "[From drawings]" in merged.raw_notes


def test_merge_readout_none_is_noop() -> None:
    """A missing readout leaves the profile unchanged."""
    profile = CaseProfile(proposal_description="Extension")
    assert merge_readout(profile, None) is profile
