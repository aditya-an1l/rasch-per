"""Smoke tests for the CLI commands (Phase 6 wiring)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from rasch_per.cli import app

runner = CliRunner()


def test_simulate_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "demo.csv"
    result = runner.invoke(
        app, ["simulate", "--n-persons", "50", "--n-items", "8", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    df = pd.read_csv(out, index_col=0)
    assert df.shape == (50, 8)


def test_validate_runs(tmp_path: Path) -> None:
    out = tmp_path / "demo.csv"
    runner.invoke(app, ["simulate", "--n-persons", "40", "--n-items", "6", "--output", str(out)])
    result = runner.invoke(app, ["validate", str(out)])
    assert result.exit_code == 0, result.output
    assert "Respondents" in result.output


def test_analyze_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "demo.csv"
    runner.invoke(app, ["simulate", "--n-persons", "120", "--n-items", "10", "--output", str(out)])
    report = tmp_path / "report.html"
    result = runner.invoke(app, ["analyze", str(out), "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert report.exists()
    assert "Internal Structure" in report.read_text(encoding="utf-8")


def test_analyze_with_groups(tmp_path: Path) -> None:
    out = tmp_path / "demo.csv"
    runner.invoke(
        app,
        ["simulate", "--n-persons", "200", "--n-items", "12", "--seed", "14", "--output", str(out)],
    )
    df = pd.read_csv(out, index_col=0)
    groups = pd.DataFrame({"group": (["ref"] * 100) + (["focal"] * 100)}, index=df.index)
    gpath = tmp_path / "groups.csv"
    groups.to_csv(gpath)
    report = tmp_path / "report_groups.html"
    result = runner.invoke(
        app,
        [
            "analyze",
            str(out),
            "--output",
            str(report),
            "--groups",
            str(gpath),
            "--dif-group",
            "group",
            "--reference",
            "ref",
            "--focal",
            "focal",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Relations to Other Variables" in report.read_text(encoding="utf-8")


def test_analyze_rejects_pdf(tmp_path: Path) -> None:
    out = tmp_path / "demo.csv"
    runner.invoke(app, ["simulate", "--n-persons", "30", "--n-items", "5", "--output", str(out)])
    result = runner.invoke(app, ["analyze", str(out), "--format", "pdf"])
    assert result.exit_code != 0
    assert "html" in result.output
