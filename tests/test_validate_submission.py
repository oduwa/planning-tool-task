import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_submission import REQUIRED_CASES, validate_submission


class ValidateSubmissionTests(unittest.TestCase):
    def _write_predictions(
        self, directory: Path, prediction: dict[str, object]
    ) -> None:
        for case_id in REQUIRED_CASES:
            (directory / f"{case_id}-prediction.json").write_text(
                json.dumps(prediction), encoding="utf-8"
            )

    def test_accepts_complete_approval_predictions(self) -> None:
        """Complete approval predictions pass validation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            predictions_dir = Path(temporary_directory)
            self._write_predictions(
                predictions_dir,
                {
                    "decision": "approve",
                    "reasons": ["The proposal complies with the relevant policies."],
                    "conditions": ["Development must begin within three years."],
                },
            )

            self.assertEqual(validate_submission(predictions_dir), [])

    def test_accepts_complete_refusal_predictions(self) -> None:
        """Complete refusal predictions pass validation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            predictions_dir = Path(temporary_directory)
            self._write_predictions(
                predictions_dir,
                {
                    "decision": "refuse",
                    "reasons": ["The proposal would harm neighbouring amenity."],
                    "conditions": [],
                },
            )

            self.assertEqual(validate_submission(predictions_dir), [])

    def test_rejects_missing_and_incomplete_predictions(self) -> None:
        """Missing files and empty required lists fail validation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            predictions_dir = Path(temporary_directory)
            (predictions_dir / "case-006-prediction.json").write_text(
                json.dumps({"decision": "approve", "reasons": [], "conditions": []}),
                encoding="utf-8",
            )

            errors = validate_submission(predictions_dir)

            self.assertIn(
                "case-006-prediction.json: 'reasons' must contain at least one item",
                errors,
            )
            self.assertIn(
                "case-006-prediction.json: 'conditions' must contain at least one item",
                errors,
            )
            self.assertIn(
                "case-007-prediction.json: required prediction file is missing", errors
            )

    def test_rejects_conditions_for_a_refusal(self) -> None:
        """Refusal predictions cannot include conditions."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            predictions_dir = Path(temporary_directory)
            self._write_predictions(
                predictions_dir,
                {
                    "decision": "refuse",
                    "reasons": ["The proposal conflicts with policy."],
                    "conditions": ["Submit details before construction."],
                },
            )

            errors = validate_submission(predictions_dir)

            self.assertIn(
                'case-006-prediction.json: "conditions" must be empty for a refusal',
                errors,
            )


if __name__ == "__main__":
    unittest.main()
