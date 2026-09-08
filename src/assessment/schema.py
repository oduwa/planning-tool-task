"""Structured schemas for the case profile and the planning assessment.

These mirror the shape of the ground-truth decisions: a set of per-theme
judgments, each citing policy and flagging whether any harm is unresolved (which
forces a refusal) or conditionable (which becomes a planning condition).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class PlanningTheme(StrEnum):
    """The material considerations an officer works through, in canonical order."""

    PRINCIPLE = "principle_of_development"
    DESIGN = "design_and_character"
    HERITAGE = "heritage"
    AMENITY = "residential_amenity"
    HIGHWAYS = "highways_and_parking"
    FLOOD = "flood_risk_and_drainage"
    ECOLOGY = "ecology_and_trees"
    CONTAMINATION = "contamination_and_environmental_health"


# Canonical ordering used when composing the reasons list.
THEME_ORDER: tuple[PlanningTheme, ...] = (
    PlanningTheme.PRINCIPLE,
    PlanningTheme.DESIGN,
    PlanningTheme.HERITAGE,
    PlanningTheme.AMENITY,
    PlanningTheme.HIGHWAYS,
    PlanningTheme.FLOOD,
    PlanningTheme.ECOLOGY,
    PlanningTheme.CONTAMINATION,
)


class HarmLevel(StrEnum):
    """Whether a theme raises harm, and if so whether it can be conditioned away."""

    NONE = "none"
    CONDITIONABLE = "conditionable"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


class DrawingRef(BaseModel):
    """A drawing reference used to build the 'approved plans' condition."""

    description: str = Field(..., description="Drawing title, e.g. 'Proposed Elevations'")
    number: str | None = Field(default=None, description="Drawing number, when stated")
    revision: str | None = Field(default=None, description="Revision letter, when stated")
    received_date: str | None = Field(
        default=None, description="Date received, when stated"
    )

    def as_clause(self) -> str:
        """Render the drawing as a single clause for the approved-plans condition."""
        parts = [self.description]
        if self.number:
            parts.append(self.number)
        if self.revision:
            parts.append(f"Revision {self.revision}")
        text = " ".join(parts)
        if self.received_date:
            text = f"{text} received {self.received_date}"
        return text


class ConsulteePosition(BaseModel):
    """A statutory/internal consultee's stance on the application."""

    body: str = Field(..., description="Consultee name, e.g. 'Highways', 'Yorkshire Water'")
    stance: str = Field(
        ..., description="One of: objection, no objection, no objection subject to conditions, comment"
    )
    summary: str = Field(..., description="Short summary of the consultee's position")


class CaseProfile(BaseModel):
    """Structured facts a decision hinges on, extracted from the pack."""

    proposal_description: str = Field(..., description="The proposed development")
    site_address: str = Field(default="", description="Site address as stated")
    postcode: str | None = Field(default=None, description="Site postcode, when stated")
    application_type: str = Field(
        default="", description="e.g. householder, full, change of use, replacement dwelling"
    )
    key_dimensions: list[str] = Field(
        default_factory=list,
        description="Measurements relevant to assessment (heights, separations, parking)",
    )
    drawing_references: list[DrawingRef] = Field(default_factory=list)
    consultee_positions: list[ConsulteePosition] = Field(default_factory=list)
    raw_notes: str = Field(default="", description="Any other salient facts")


class PlanReadout(BaseModel):
    """What a vision pass recovered from the drawings."""

    key_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions read from plans: ridge/eaves heights, storeys, separations",
    )
    drawing_references: list[DrawingRef] = Field(
        default_factory=list,
        description="Drawing numbers/revisions/dates visible in title blocks",
    )
    layout_notes: str = Field(
        default="", description="Layout, orientation, parking, boundary observations"
    )


class ThemeAssessment(BaseModel):
    """The officer-style judgment for one material consideration."""

    theme: PlanningTheme
    finding: str = Field(
        ...,
        description=(
            "A self-contained planning reason paragraph in the style of a decision "
            "notice, citing the specific Local Plan policies and NPPF paragraphs relied on."
        ),
    )
    harm: HarmLevel
    cited_policies: list[str] = Field(default_factory=list)
    suggested_condition: str | None = Field(
        default=None,
        description="When harm is conditionable, a condition ending with a 'Reason:' clause",
    )


class CaseAssessment(BaseModel):
    """The full set of per-theme judgments for a case."""

    themes: list[ThemeAssessment] = Field(default_factory=list)

    def unresolved(self) -> list[ThemeAssessment]:
        """Return themes carrying unresolved (non-conditionable) harm."""
        return [t for t in self.themes if t.harm == HarmLevel.UNRESOLVED]
