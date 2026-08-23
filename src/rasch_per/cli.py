"""Command-line interface for rasch-per.

The CLI is a thin wrapper: every command delegates to a documented,
directly-importable Python function. Logic lives in the library modules, not
here.

Commands:
    analyze   - run the full validity pipeline and write an HTML/PDF report
    simulate  - generate a synthetic response CSV for trying out the tool
    validate  - data validation/diagnostics only (missingness, value range)
"""

from __future__ import annotations

from pathlib import Path

import typer

from rasch_per import __version__

app = typer.Typer(
    name="rasch-per",
    help="Rasch model and CTT psychometric analysis for education research.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rasch-per {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
) -> None:
    """Rasch model and CTT psychometric analysis for education research."""


@app.command()
def analyze(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Response CSV path."),
    output: Path = typer.Option(Path("report.html"), help="Output report path."),
    fmt: str = typer.Option("html", "--format", help="Output format: html or pdf."),
    estimator: str = typer.Option("MML", help="Rasch estimation method: MML or JML."),
    groups: Path | None = typer.Option(None, help="Optional CSV of person_id -> group metadata."),
    dif_group: str | None = typer.Option(None, help="Column in --groups to use for DIF."),
    fit_bounds: str = typer.Option("low_stakes", help="low_stakes | high_stakes | lo,hi"),
    min_response_rate: float = typer.Option(0.5, help="Drop respondents below this rate."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Run the full validity analysis pipeline and write a report."""
    _stub("analyze is implemented in Phase 5/6")


@app.command()
def simulate(
    n_persons: int = typer.Option(300, help="Number of simulated persons."),
    n_items: int = typer.Option(20, help="Number of simulated items."),
    seed: int | None = typer.Option(None, help="Random seed for reproducibility."),
    output: Path = typer.Option(Path("demo.csv"), help="Output CSV path."),
) -> None:
    """Generate a synthetic dichotomous response CSV."""
    _stub("simulate is implemented in Phase 2")


@app.command()
def validate(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Response CSV path."),
) -> None:
    """Run data validation/diagnostics without the full analysis."""
    _stub("validate is implemented in Phase 6")


def _stub(message: str) -> None:
    typer.secho(f"Not yet implemented: {message}", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
