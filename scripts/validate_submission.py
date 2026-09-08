#!/usr/bin/env python3
"""Validate the prediction files required for a candidate submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_CASES = tuple(f"case-{number:03d}" for number in range(6, 11))
REQUIRED_FIELDS = {"decision", "reasons", "conditions"}
VALID_DECISIONS = {"approve", "refuse"}


class DuplicateFieldError(ValueError):
    """Raised when a JSON object contains the same field more than once."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateFieldError(f"duplicate field {key!r}")
        result[key] = value
    return result


def _validate_text_list(
    value: Any,
    *,
    field: str,
    file_name: str,
    required: bool,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list):
        return [f"{file_name}: {field!r} must be an array of strings"]
    if required and not value:
        errors.append(f"{file_name}: {field!r} must contain at least one item")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{file_name}: {field}[{index}] must be a non-empty string")
    return errors


def validate_prediction(path: Path) -> list[str]:
    """Return all validation errors for one prediction file."""
    try:
        prediction = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateFieldError) as error:
        return [f"{path.name}: invalid JSON ({error})"]

    if not isinstance(prediction, dict):
        return [f"{path.name}: the top-level JSON value must be an object"]

    errors: list[str] = []
    fields = set(prediction)
    missing = sorted(REQUIRED_FIELDS - fields)
    unexpected = sorted(fields - REQUIRED_FIELDS)
    if missing:
        errors.append(f"{path.name}: missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{path.name}: unexpected fields: {', '.join(unexpected)}")

    decision = prediction.get("decision")
    if decision not in VALID_DECISIONS:
        errors.append(f'{path.name}: "decision" must be either "approve" or "refuse"')

    errors.extend(
        _validate_text_list(
            prediction.get("reasons"),
            field="reasons",
            file_name=path.name,
            required=True,
        )
    )
    errors.extend(
        _validate_text_list(
            prediction.get("conditions"),
            field="conditions",
            file_name=path.name,
            required=decision == "approve",
        )
    )

    conditions = prediction.get("conditions")
    if decision == "refuse" and isinstance(conditions, list) and conditions:
        errors.append(f'{path.name}: "conditions" must be empty for a refusal')

    return errors


def validate_submission(predictions_dir: Path) -> list[str]:
    """Return all validation errors across the required prediction files."""
    errors: list[str] = []
    for case_id in REQUIRED_CASES:
        path = predictions_dir / f"{case_id}-prediction.json"
        if not path.is_file():
            errors.append(f"{path.name}: required prediction file is missing")
            continue
        errors.extend(validate_prediction(path))
    return errors


def main() -> int:
    """Validate predictions from the repository root and report the result."""
    repository_root = Path(__file__).resolve().parent.parent
    errors = validate_submission(repository_root / "predictions")
    if errors:
        message = "\n".join(
            ["Submission validation failed:", *(f"- {error}" for error in errors)]
        )
        sys.stderr.write(f"{message}\n")
        return 1

    sys.stdout.write("Submission is valid: all five prediction files are complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
