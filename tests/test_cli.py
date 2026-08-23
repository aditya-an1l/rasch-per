"""Tests for the CLI entry point (typer)."""

from __future__ import annotations

from typer.testing import CliRunner

from rasch_per import __version__
from rasch_per.cli import app

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Rasch model and CTT" in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_commands_listed() -> None:
    result = runner.invoke(app, ["--help"])
    for command in ("analyze", "simulate", "validate"):
        assert command in result.output


def test_stub_command_exit_code() -> None:
    # analyze on a missing file fails arg validation; use validate on a real
    # path instead - it is a stub until Phase 6 and must exit non-zero with a message.
    import pathlib

    tmp = pathlib.Path("tests/data/synthetic_small.csv")
    tmp.touch(exist_ok=True)
    result = runner.invoke(app, ["validate", str(tmp)])
    assert result.exit_code == 2
    assert "Not yet implemented" in (result.output + str(result.exception or ""))
