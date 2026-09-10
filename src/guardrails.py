"""Substantive, pipeline-wired guardrails for the planning decision tool.

These are not cosmetic string checks. Each guardrail reads real pipeline state —
the extracted profile, the per-theme assessment, the emitted decision, and the
live policy index — and can escalate a draft for mandatory human review:

- **scope**: is this actually an assessable housing application with the minimum
  facts needed to decide? Refuse to pretend otherwise.
- **grounding**: does every policy the decision cites actually exist in the
  corpus? Cited references are verified against the policy index, catching
  hallucinated citations — a refusal resting on an invented policy is a blocker.
- **consistency**: are the structural invariants of a decision notice satisfied
  (refusals carry no conditions, approvals carry the standard conditions, the
  refuse/approve outcome matches the unresolved-harm count)?
- **approval_bias**: the domain-specific safeguard against the well-documented
  tendency of planning LLMs to approve. An outstanding statutory objection that
  the assessment did not turn into refusal or a condition blocks auto-drafting.
- **abstention**: too little evidence (no citations anywhere, or an all-clear
  assessment on a contested pack) escalates for review rather than rubber-stamping.

The tool produces *draft* decisions, so the active intervention is escalation:
blockers set ``requires_human_review`` and are surfaced in the officer draft and
the evaluation trace. The binary approve/refuse prediction itself is left intact.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from assessment.schema import CaseAssessment, CaseProfile, HarmLevel
from decision import Decision, DecisionType
from policy.retriever import PolicyRetriever

# References shorter than this are too vague to verify (e.g. a bare "Policy").
_MIN_REF_LENGTH = 4


class Severity(StrEnum):
    """How serious a guardrail finding is."""

    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class GuardrailFinding(BaseModel):
    """A single guardrail observation."""

    check: str = Field(..., description="Which guardrail produced this finding")
    severity: Severity
    message: str


class GuardrailReport(BaseModel):
    """The aggregated result of running every guardrail on a case."""

    findings: list[GuardrailFinding] = Field(default_factory=list)

    @property
    def blockers(self) -> list[GuardrailFinding]:
        """Return findings that block unattended auto-drafting."""
        return [f for f in self.findings if f.severity == Severity.BLOCKER]

    @property
    def warnings(self) -> list[GuardrailFinding]:
        """Return advisory warning findings."""
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def requires_human_review(self) -> bool:
        """Return True when any blocker means an officer must review before issue."""
        return bool(self.blockers)

    def to_markdown(self) -> str:
        """Render the report as a markdown section for the officer draft."""
        if not self.findings:
            return "## Guardrails\n\nAll automated guardrails passed.\n"
        header = (
            "## Guardrails\n\n"
            f"**Human review required: {'YES' if self.requires_human_review else 'no'}**\n\n"
        )
        rows = "\n".join(
            f"- **{f.severity.value.upper()}** ({f.check}): {f.message}"
            for f in self.findings
        )
        return header + rows + "\n"


def _looks_like_citation(reference: str) -> bool:
    """Return True if a string looks like a verifiable policy/paragraph citation.

    Args:
        reference: A cited-policy string from the assessment.
    """
    text = reference.strip().lower()
    if len(text) < _MIN_REF_LENGTH:
        return False
    return "policy" in text or "paragraph" in text or "para" in text or "nppf" in text


def check_scope(profile: CaseProfile) -> list[GuardrailFinding]:
    """Confirm the case is an assessable housing application with usable facts.

    Args:
        profile: The extracted case profile.
    """
    findings: list[GuardrailFinding] = []
    if not profile.proposal_description.strip():
        findings.append(
            GuardrailFinding(
                check="scope",
                severity=Severity.BLOCKER,
                message="No proposal description was extracted; the pack cannot be assessed.",
            )
        )
    if not profile.postcode:
        findings.append(
            GuardrailFinding(
                check="scope",
                severity=Severity.INFO,
                message="No postcode found; site constraints could not be verified geospatially.",
            )
        )
    out_of_scope = ("advertisement", "telecommunications", "industrial estate", "quarry")
    haystack = f"{profile.application_type} {profile.proposal_description}".lower()
    if any(term in haystack for term in out_of_scope):
        findings.append(
            GuardrailFinding(
                check="scope",
                severity=Severity.WARNING,
                message="Proposal may fall outside householder/housing scope; confirm the case type.",
            )
        )
    return findings


def check_grounding(
    assessment: CaseAssessment, retriever: PolicyRetriever
) -> list[GuardrailFinding]:
    """Verify every cited policy exists in the corpus; flag unsupported findings.

    Args:
        assessment: The per-theme assessment.
        retriever: Loaded policy retriever (source of truth for existence).
    """
    findings: list[GuardrailFinding] = []
    for theme in assessment.themes:
        citations = [c for c in theme.cited_policies if _looks_like_citation(c)]
        material = theme.harm in {HarmLevel.UNRESOLVED, HarmLevel.CONDITIONABLE}

        if material and not citations:
            findings.append(
                GuardrailFinding(
                    check="grounding",
                    severity=Severity.WARNING,
                    message=f"Theme '{theme.theme.value}' asserts {theme.harm.value} harm without citing any policy.",
                )
            )
        for citation in citations:
            if retriever.verify_citation(citation):
                continue
            severity = (
                Severity.BLOCKER if theme.harm == HarmLevel.UNRESOLVED else Severity.WARNING
            )
            findings.append(
                GuardrailFinding(
                    check="grounding",
                    severity=severity,
                    message=(
                        f"Theme '{theme.theme.value}' cites '{citation}', which was not "
                        "found in the policy corpus (possible hallucinated citation)."
                    ),
                )
            )
    return findings


def check_consistency(
    decision: Decision, assessment: CaseAssessment
) -> list[GuardrailFinding]:
    """Check the emitted decision satisfies decision-notice structural invariants.

    Args:
        decision: The synthesized decision.
        assessment: The per-theme assessment it was derived from.
    """
    findings: list[GuardrailFinding] = []
    unresolved = assessment.unresolved()

    if not decision.reasons:
        findings.append(
            GuardrailFinding(
                check="consistency",
                severity=Severity.BLOCKER,
                message="Decision has no reasons.",
            )
        )
    if decision.decision == DecisionType.REFUSE:
        if decision.conditions:
            findings.append(
                GuardrailFinding(
                    check="consistency",
                    severity=Severity.BLOCKER,
                    message="Refusal must not carry conditions.",
                )
            )
        if not unresolved:
            findings.append(
                GuardrailFinding(
                    check="consistency",
                    severity=Severity.BLOCKER,
                    message="Decision is refuse but no theme carries unresolved harm.",
                )
            )
    else:
        if unresolved:
            findings.append(
                GuardrailFinding(
                    check="consistency",
                    severity=Severity.BLOCKER,
                    message=(
                        f"Decision is approve but {len(unresolved)} theme(s) carry "
                        "unresolved harm that should force refusal."
                    ),
                )
            )
        if len(decision.conditions) < 2:
            findings.append(
                GuardrailFinding(
                    check="consistency",
                    severity=Severity.WARNING,
                    message="Approval is missing the standard commencement/approved-plans conditions.",
                )
            )
    return findings


def check_approval_bias(
    profile: CaseProfile, assessment: CaseAssessment, decision: Decision
) -> list[GuardrailFinding]:
    """Block silent approvals that leave a statutory objection unresolved.

    Args:
        profile: The extracted case profile (source of consultee stances).
        assessment: The per-theme assessment.
        decision: The synthesized decision.
    """
    findings: list[GuardrailFinding] = []

    objectors = [
        c
        for c in profile.consultee_positions
        if "object" in c.stance.lower() and "no objection" not in c.stance.lower()
    ]
    if objectors and decision.decision == DecisionType.APPROVE:
        assessed_text = " ".join(t.finding.lower() for t in assessment.themes)
        for objector in objectors:
            engaged = objector.body.lower() in assessed_text
            findings.append(
                GuardrailFinding(
                    check="approval_bias",
                    severity=Severity.WARNING if engaged else Severity.BLOCKER,
                    message=(
                        f"{objector.body} objected but the decision is approve"
                        + (
                            "; confirm the objection is properly resolved by condition."
                            if engaged
                            else " and the assessment does not engage the objection."
                        )
                    ),
                )
            )

    assessable = [t for t in assessment.themes if t.harm != HarmLevel.NOT_APPLICABLE]
    if (
        decision.decision == DecisionType.APPROVE
        and assessable
        and all(t.harm == HarmLevel.NONE for t in assessable)
    ):
        findings.append(
            GuardrailFinding(
                check="approval_bias",
                severity=Severity.INFO,
                message="Every theme was found entirely harm-free; sanity-check this is not over-optimistic.",
            )
        )
    return findings


def check_abstention(assessment: CaseAssessment) -> list[GuardrailFinding]:
    """Escalate when there is too little cited evidence to stand behind a decision.

    Args:
        assessment: The per-theme assessment.
    """
    total_citations = sum(len(t.cited_policies) for t in assessment.themes)
    if not assessment.themes:
        return [
            GuardrailFinding(
                check="abstention",
                severity=Severity.BLOCKER,
                message="The assessment produced no themes; insufficient basis for a decision.",
            )
        ]
    if total_citations == 0:
        return [
            GuardrailFinding(
                check="abstention",
                severity=Severity.BLOCKER,
                message="No policy was cited anywhere in the assessment; escalate for review.",
            )
        ]
    return []


def run_guardrails(
    profile: CaseProfile,
    assessment: CaseAssessment,
    decision: Decision,
    retriever: PolicyRetriever,
) -> GuardrailReport:
    """Run every guardrail and aggregate the findings into one report.

    Args:
        profile: The extracted case profile.
        assessment: The per-theme assessment.
        decision: The synthesized decision.
        retriever: Loaded policy retriever, for citation verification.
    """
    findings: list[GuardrailFinding] = []
    findings += check_scope(profile)
    findings += check_grounding(assessment, retriever)
    findings += check_consistency(decision, assessment)
    findings += check_approval_bias(profile, assessment, decision)
    findings += check_abstention(assessment)
    return GuardrailReport(findings=findings)
