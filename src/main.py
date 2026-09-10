"""Command-line entry point for the planning decision-support tool."""

from __future__ import annotations

import asyncio

import typer
from loguru import logger

from llm import ModelConfig
from paths import POLICY_INDEX_DIR
from pipeline import run_case_detailed, write_outputs
from policy.index import build_policy_index
from policy.retriever import PolicyRetriever

app = typer.Typer(
    name="main",
    help="Augmented planning decision-making",
    add_completion=True,
    no_args_is_help=True,
)

HOLDOUT_CASES = tuple(f"case-{number:03d}" for number in range(6, 11))


@app.command()
def build_index(
    force: bool = typer.Option(False, help="Rebuild even if an index already exists."),
) -> None:
    """Build the one-time policy embedding index from the data directory."""
    cfg = ModelConfig.from_env()
    count = asyncio.run(build_policy_index(cfg, POLICY_INDEX_DIR, force=force))
    typer.echo(f"Policy index ready: {count} chunks.")


@app.command()
def generate_planning_assessment(
    application_name: str = typer.Argument(
        ..., help="The name of the application to evaluate, e.g. case-006."
    ),
) -> None:
    """Generate a draft response for a single planning application."""
    cfg = ModelConfig.from_env()
    retriever = PolicyRetriever(cfg)
    result = asyncio.run(run_case_detailed(application_name, cfg, retriever))
    output = write_outputs(application_name, result)
    typer.echo(
        result.decision.to_markdown(
            batch_name=application_name, local_authority="Doncaster Council"
        )
    )
    typer.echo(f"\n{result.guardrails.to_markdown()}")
    if result.guardrails.requires_human_review:
        typer.echo("NOTE: guardrails flagged this draft for mandatory officer review.")
    typer.echo(f"\nPrediction written to {output}")


@app.command()
def batch(
    cases: list[str] = typer.Argument(
        None, help="Cases to run. Defaults to the five holdout cases (006-010)."
    ),
) -> None:
    """Run the pipeline across multiple cases (defaults to the holdout set)."""
    cfg = ModelConfig.from_env()
    retriever = PolicyRetriever(cfg)
    targets = tuple(cases) if cases else HOLDOUT_CASES

    async def _run_all() -> None:
        for case_id in targets:
            try:
                result = await run_case_detailed(case_id, cfg, retriever)
                output = write_outputs(case_id, result)
                review = " [REVIEW]" if result.guardrails.requires_human_review else ""
                typer.echo(f"{case_id}: {result.decision.decision.value}{review} -> {output}")
            except Exception as error:  # noqa: BLE001 - keep the batch going
                logger.error(f"{case_id} failed: {error}")

    asyncio.run(_run_all())


if __name__ == "__main__":
    app()
