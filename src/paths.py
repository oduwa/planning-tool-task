"""Canonical filesystem locations for the planning decision-support tool.

`src/` is mapped to the package root (see `pyproject.toml`), so this module's
parent directory is `src/` and its grandparent is the repository root.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
APPLICATIONS_DIR = DATA_DIR / "doncaster" / "applications"
DECISIONS_DIR = DATA_DIR / "doncaster" / "decisions"

# Where `uv run parser` writes mirrored, pre-extracted PDF JSON.
EXTRACTED_DIR = DATA_DIR / "extracted"

NATIONAL_POLICY_DIR = DATA_DIR / "national_policy"
LOCAL_POLICY_DIR = DATA_DIR / "local_policy"
SPDS_DIR = LOCAL_POLICY_DIR / "spds"

PREDICTIONS_DIR = REPO_ROOT / "predictions"

# Cached, one-time policy embedding index (git-ignored).
POLICY_INDEX_DIR = REPO_ROOT / ".policy_index"

# Human-readable officer drafts produced alongside the JSON predictions.
DRAFTS_DIR = REPO_ROOT / "drafts"


def application_dir(case_id: str) -> Path:
    """Return the application-pack directory for a case id such as ``case-006``.

    Args:
        case_id: Case identifier matching a folder under the applications dir.
    """
    return APPLICATIONS_DIR / case_id


def decision_path(case_id: str) -> Path:
    """Return the ground-truth ``decision.json`` path for a training case.

    Args:
        case_id: Case identifier such as ``case-001``.
    """
    return DECISIONS_DIR / case_id / "decision.json"


def prediction_path(case_id: str) -> Path:
    """Return the output prediction path for a case, e.g. ``case-006``.

    Args:
        case_id: Case identifier such as ``case-006``.
    """
    return PREDICTIONS_DIR / f"{case_id}-prediction.json"


def extracted_json_path(pdf_path: Path) -> Path:
    """Return the cached JSON path that ``uv run parser`` writes for a PDF.

    The batch parser mirrors the ``data/`` tree under ``data/extracted`` and
    swaps the ``.pdf`` suffix for ``.json``.

    Args:
        pdf_path: Path to a source PDF located under the data directory.

    Raises:
        ValueError: If ``pdf_path`` is not located under the data directory.
    """
    relative = pdf_path.resolve().relative_to(DATA_DIR.resolve())
    return EXTRACTED_DIR / relative.with_suffix(".json")
