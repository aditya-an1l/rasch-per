"""Tests for the full-pipeline HTML report (Phase 5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")


from rasch_per.report import generate_report  # noqa: E402
from rasch_per.simulate import simulate_rasch_data  # noqa: E402


def test_generate_report_writes_html(tmp_path: Path) -> None:
    df = simulate_rasch_data(n_persons=200, n_items=10, seed=7)
    out = tmp_path / "report.html"
    generate_report(df, output=out)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<html" in html
    # Internal Structure section is always present.
    assert "Internal Structure" in html
    # Plots are embedded as base64 PNGs.
    assert "data:image/png;base64," in html
    # No groups -> no Relations section.
    assert "Relations to Other Variables" not in html


def test_generate_report_with_groups_includes_relations(tmp_path: Path) -> None:
    n = 400
    groups = ["ref"] * (n // 2) + ["focal"] * (n // 2)
    df = simulate_rasch_data(
        n_persons=n,
        n_items=12,
        seed=14,
        groups=groups,
        focal_label="focal",
        dif_effects={0: 0.9},
    )
    out = tmp_path / "report_groups.html"
    generate_report(df, output=out, groups=groups, reference="ref", focal="focal")
    html = out.read_text(encoding="utf-8")
    assert "Relations to Other Variables" in html
    assert "DIF" in html


def test_generate_report_accepts_file_object(tmp_path: Path) -> None:
    df = simulate_rasch_data(n_persons=120, n_items=8, seed=9)
    target = tmp_path / "out.html"
    with target.open("w", encoding="utf-8") as fh:
        generate_report(df, output=fh)
    assert target.exists()
    assert "Internal Structure" in target.read_text(encoding="utf-8")
