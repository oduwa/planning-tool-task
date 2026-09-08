"""
Iterate over parsed Doncaster delegated report JSON files.

By default this parses each delegated report with the OpenAI Agents SDK and
prints structured Decision JSON. Use ``--list`` to print report paths without
calling the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from decision import Decision

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = (
    REPO_ROOT / "data" / "extracted" / "doncaster" / "delegated_reports"
)
DEFAULT_PROMPT_PATH = Path(__file__).with_name("reasons-prompt.md")
ReasoningLevel = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_LEVELS: tuple[ReasoningLevel, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


def iter_report_paths(reports_dir: Path) -> list[Path]:
    """Return parsed delegated report JSON paths in stable reference order."""
    return sorted(
        reports_dir.glob("*/*.json"), key=lambda path: (path.parent.name, path.name)
    )


def resolve_report_paths(reports_dir: Path, report_path: Path | None) -> list[Path]:
    """Resolve either one named report JSON or all reports in a directory."""
    if report_path is not None:
        return [report_path]

    return iter_report_paths(reports_dir)


def strip_image_bytes(value: Any) -> Any:
    """Remove large image byte payloads while preserving the report JSON shape."""
    if isinstance(value, dict):
        return {
            key: strip_image_bytes(item)
            for key, item in value.items()
            if key != "image_bytes"
        }

    if isinstance(value, list):
        return [strip_image_bytes(item) for item in value]

    return value


def order_report_json(report_json: dict[str, Any]) -> dict[str, Any]:
    """Sort pages and blocks into document order for stable model input."""
    ordered = strip_image_bytes(report_json)
    pages = sorted(
        report_json.get("pages", []), key=lambda page: page.get("page_number", 0)
    )
    ordered_pages = []

    for page in pages:
        ordered_page = strip_image_bytes(page)
        ordered_page["blocks"] = sorted(
            ordered_page.get("blocks", []),
            key=lambda block: block.get("block_index", 0),
        )
        ordered_pages.append(ordered_page)

    ordered["pages"] = ordered_pages
    return ordered


def load_report(path: Path) -> dict[str, Any]:
    """Load and normalize one delegated report JSON."""
    report_json = json.loads(path.read_text(encoding="utf-8"))
    return order_report_json(report_json)


def build_agent_input(path: Path) -> str:
    """Build the user input sent to the decision parsing agent."""
    payload = {
        "reference": path.parent.name,
        "delegated_report_json": load_report(path),
    }
    return json.dumps(payload, ensure_ascii=False)


def write_decision(output_dir: Path, reference: str, decision: Decision) -> Path:
    """Write a parsed decision under the reference folder."""
    decision_dir = output_dir / reference
    decision_dir.mkdir(parents=True, exist_ok=True)
    output_path = decision_dir / "decision.json"
    output_path.write_text(
        f"{json.dumps(decision.model_dump(mode='json'), indent=2)}\n",
        encoding="utf-8",
    )
    sys.stderr.write(f"Wrote parsed decision for {reference}: {output_path}\n")
    return output_path


def build_model_settings(reasoning_level: ReasoningLevel | None) -> Any:
    """Build Agents SDK model settings, including optional reasoning effort."""
    from agents import ModelSettings
    from openai.types.shared.reasoning import Reasoning

    if reasoning_level is None:
        return ModelSettings()

    return ModelSettings(reasoning=Reasoning(effort=reasoning_level))


async def parse_report(
    path: Path,
    instructions: str,
    model: str,
    reasoning_level: ReasoningLevel | None,
) -> Decision:
    """Run the Agents SDK over one delegated report JSON."""
    try:
        from agents import Agent, Runner
    except ImportError as exc:
        msg = "OpenAI Agents SDK is not installed. Install it with `uv add openai-agents`."
        raise RuntimeError(msg) from exc

    agent = Agent(
        name="Planning decision parser",
        instructions=instructions,
        model=model,
        model_settings=build_model_settings(reasoning_level),
        output_type=Decision,
    )
    result = await Runner.run(agent, build_agent_input(path))

    if isinstance(result.final_output, Decision):
        return result.final_output

    return Decision.model_validate(result.final_output)


async def parse_reports(
    paths: list[Path],
    instructions: str,
    model: str,
    reasoning_level: ReasoningLevel | None,
    output_dir: Path | None,
) -> list[dict[str, Any]]:
    """Parse reports sequentially and return serializable records."""
    records: list[dict[str, Any]] = []

    for path in paths:
        decision = await parse_report(path, instructions, model, reasoning_level)
        record: dict[str, Any] = {
            "reference": path.parent.name,
            "report_path": str(path),
            "decision": decision.model_dump(mode="json"),
        }

        if output_dir is not None:
            record["output_path"] = str(
                write_decision(output_dir, path.parent.name, decision)
            )

        records.append(record)

    return records


async def async_main() -> None:
    """Run the delegated report iterator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory containing parsed delegated report JSON folders.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path to one parsed delegated report JSON file to process.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Markdown instructions for the decision parsing agent.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        help="OpenAI model to use via the Agents SDK.",
    )
    parser.add_argument(
        "--reasoning-level",
        choices=REASONING_LEVELS,
        default=os.getenv("OPENAI_REASONING_LEVEL", "medium"),
        help="Optional reasoning effort to pass to the model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where parsed decision.json files should be written.",
    )
    args = parser.parse_args()

    paths = resolve_report_paths(args.reports_dir, args.report_path)

    instructions = args.prompt_path.read_text(encoding="utf-8")
    records = await parse_reports(
        paths,
        instructions,
        args.model,
        args.reasoning_level,
        args.output_dir,
    )
    sys.stdout.write(f"{json.dumps(records, indent=2)}\n")


def main() -> None:
    """CLI entrypoint."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
