"""Deterministic assembly of a Decision from the per-theme assessment.

The decision boundary is deterministic and auditable: refuse if any theme
carries unresolved harm, otherwise approve with conditions. Reasons are the
per-theme findings in canonical order; approvals additionally carry the two
standard conditions (commencement time limit and approved plans) followed by any
theme-specific conditions.
"""

from __future__ import annotations

from assessment.schema import (
    THEME_ORDER,
    CaseAssessment,
    CaseProfile,
    HarmLevel,
    ThemeAssessment,
)
from decision import Decision, DecisionType

_TIME_LIMIT_CONDITION = (
    "The development must be begun not later than the expiration of three years "
    "beginning with the date of this permission. Reason: Condition required to be "
    "imposed by Section 91 (as amended) of the Town and Country Planning Act 1990."
)


def _ordered_themes(assessment: CaseAssessment) -> list[ThemeAssessment]:
    """Return themes in canonical order, appending any unrecognized extras.

    Args:
        assessment: The per-theme assessment.
    """
    order = {theme: index for index, theme in enumerate(THEME_ORDER)}
    return sorted(assessment.themes, key=lambda t: order.get(t.theme, len(order)))


def _approved_plans_condition(profile: CaseProfile) -> str:
    """Build the 'in accordance with the approved plans' condition.

    Args:
        profile: The case profile carrying drawing references.
    """
    base = (
        "The development hereby permitted must be carried out and completed entirely "
        "in accordance with the terms of this permission and the approved plans"
    )
    if profile.drawing_references:
        clauses = "; ".join(ref.as_clause() for ref in profile.drawing_references)
        base = f"{base}: {clauses}"
    return (
        f"{base}. Reason: To ensure that the development is carried out in accordance "
        "with the application as approved."
    )


def synthesize_decision(
    assessment: CaseAssessment, profile: CaseProfile
) -> Decision:
    """Assemble a final Decision from the assessment and profile.

    Args:
        assessment: The per-theme assessment.
        profile: The case profile (for drawing references and conditions).
    """
    themes = _ordered_themes(assessment)
    refuse = any(t.harm == HarmLevel.UNRESOLVED for t in themes)

    reasons = [t.finding.strip() for t in themes if t.finding.strip()]

    if refuse:
        return Decision(
            decision=DecisionType.REFUSE,
            reasons=reasons,
            conditions=[],
        )

    conditions = [_TIME_LIMIT_CONDITION, _approved_plans_condition(profile)]
    for theme in themes:
        if theme.harm == HarmLevel.CONDITIONABLE and theme.suggested_condition:
            conditions.append(theme.suggested_condition.strip())

    return Decision(
        decision=DecisionType.APPROVE,
        reasons=reasons,
        conditions=conditions,
    )
