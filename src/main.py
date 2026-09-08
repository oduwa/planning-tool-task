import typer

app = typer.Typer(
    name="main",
    help="Augmented planning decision-making",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def generate_planning_assessment(
    application_name: str = typer.Argument(
        ..., help="The name of the application to evaluate."
    ),
) -> None:
    """Generate a draft response for a planning application."""
    # TODO: your code here

    # from decision import Decision
    # decision: Decision = method()
    # decision.to_markdown()


if __name__ == "__main__":
    app()
