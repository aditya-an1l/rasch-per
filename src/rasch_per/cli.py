"""Command-line interface for rasch-per.

The CLI is a thin wrapper: every command delegates to a documented,
directly-importable Python function. Logic lives in the library modules, not
here.

Commands:
    analyze   - run the full validity pipeline and write an HTML report
    simulate  - generate a synthetic response CSV for trying out the tool
    validate  - data validation/diagnostics only (missingness, value range)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from rasch_per import __version__
from rasch_per.data import ResponseData
from rasch_per.report import generate_report
from rasch_per.simulate import simulate_rasch_data

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
    fmt: str = typer.Option("html", "--format", help="Output format: html."),
    estimator: str = typer.Option("MML", help="Rasch estimation method: MML or JML."),
    groups: Path | None = typer.Option(None, help="Optional CSV of person_id -> group metadata."),
    dif_group: str | None = typer.Option(None, help="Column in --groups to use for DIF."),
    reference: str | None = typer.Option(None, help="Reference group label for DIF."),
    focal: str | None = typer.Option(None, help="Focal group label for DIF."),
    min_response_rate: float = typer.Option(0.5, help="Drop respondents below this rate."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Run the full validity analysis pipeline and write a report."""
    if fmt != "html":
        raise typer.BadParameter("Only '--format html' is supported in this release.")
    if (groups is None) != (dif_group is None):
        raise typer.BadParameter("--groups and --dif-group must be supplied together.")
    if (reference is None) != (focal is None):
        raise typer.BadParameter("--reference and --focal must be supplied together.")

    df = pd.read_csv(csv_path, index_col=0)
    data = ResponseData(df).filter_min_response_rate(min_response_rate)
    if verbose:
        typer.echo(f"Loaded {data.n_persons} persons x {data.n_items} items")

    groups_arg: object = None
    ref_arg: str | None = reference
    foc_arg: str | None = focal
    if groups is not None and dif_group is not None:
        gdf = pd.read_csv(groups, index_col=0)
        if dif_group not in gdf.columns:
            raise typer.BadParameter(f"Column {dif_group!r} not found in --groups file")
        labels = gdf[dif_group].reindex(data.person_ids)
        if reference is None or focal is None:
            uniq = [str(u) for u in labels.dropna().unique()]
            if len(uniq) < 2:
                raise typer.BadParameter("--groups must contain at least two group labels")
            ref_arg = reference or uniq[0]
            foc_arg = focal or uniq[1]
        groups_arg = labels.to_numpy()

    generate_report(
        data.to_dataframe(),
        output=output,
        title="Rasch/CTT Validity Report",
        groups=groups_arg,
        reference=ref_arg,
        focal=foc_arg,
        estimator=estimator,
    )
    if verbose:
        typer.echo(f"Wrote {output}")


@app.command()
def simulate(
    n_persons: int = typer.Option(300, help="Number of simulated persons."),
    n_items: int = typer.Option(20, help="Number of simulated items."),
    seed: int | None = typer.Option(None, help="Random seed for reproducibility."),
    output: Path = typer.Option(Path("demo.csv"), help="Output CSV path."),
) -> None:
    """Generate a synthetic dichotomous response CSV."""
    result = simulate_rasch_data(n_persons=n_persons, n_items=n_items, seed=seed)
    df = result[0] if isinstance(result, tuple) else result
    df = df.set_index("person_id") if "person_id" in df.columns else df
    df.to_csv(output)
    typer.echo(f"Wrote {output} ({len(df)} persons x {df.shape[1]} items)")


@app.command()
def validate(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="Response CSV path."),
) -> None:
    """Run data validation/diagnostics without the full analysis."""
    df = pd.read_csv(csv_path, index_col=0)
    data = ResponseData(df)
    typer.echo(f"Respondents: {data.n_persons}  Items: {data.n_items}")
    typer.echo("\nMissingness per item (%):")
    typer.echo(data.missing_by_item().to_string())
    person_missing = data.missing_by_person()
    typer.echo(
        "\nPerson missingness: "
        f"min={person_missing.min():.1f}%  median={person_missing.median():.1f}%  "
        f"max={person_missing.max():.1f}%"
    )
    values = data.to_numpy()
    typer.echo(
        f"\nResponse values: min={np.nanmin(values):.0f}  max={np.nanmax(values):.0f} "
        "(expected 0/1 with possible NaN for missing)"
    )


if __name__ == "__main__":
    app()
