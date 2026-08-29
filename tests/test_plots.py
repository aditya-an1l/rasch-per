"""Tests for matplotlib plotting functions (Phase 5).

Every plot function must return a matplotlib Figure without calling plt.show.
Follows the reference-value-testing skill: structural checks only (axes, data
presence), since pixel output is not asserted.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from rasch_per import DIFAnalysis, RaschModel  # noqa: E402
from rasch_per.ctt import CTTAnalysis  # noqa: E402
from rasch_per.data import ResponseData  # noqa: E402
from rasch_per.dif import benjamini_hochberg  # noqa: E402
from rasch_per.plots import (  # noqa: E402
    plot_dif_contrasts,
    plot_group_ability_distributions,
    plot_icc,
    plot_item_difficulty_bar,
    plot_item_discrimination_bar,
    plot_test_information,
    plot_wright_map,
)
from rasch_per.simulate import simulate_rasch_data  # noqa: E402


@pytest.fixture()
def fitted():
    df = simulate_rasch_data(n_persons=200, n_items=10, seed=7)
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    ctt = CTTAnalysis(ResponseData(df)).run()
    return df, model, ctt


def test_plot_icc_returns_figure(fitted) -> None:
    _, model, _ = fitted
    fig = plot_icc(model, "item_1")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) >= 1
    plt.close(fig)


def test_plot_wright_map(fitted) -> None:
    _, model, _ = fitted
    fig = plot_wright_map(model)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_test_information(fitted) -> None:
    _, model, _ = fitted
    fig = plot_test_information(model)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ctt_bars(fitted) -> None:
    _, _, ctt = fitted
    d = plot_item_difficulty_bar(ctt)
    r = plot_item_discrimination_bar(ctt)
    assert isinstance(d, plt.Figure)
    assert isinstance(r, plt.Figure)
    plt.close(d)
    plt.close(r)


def test_plot_dif_contrasts() -> None:
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
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    dif = DIFAnalysis(model, groups, reference="ref", focal="focal").analyze()
    fig = plot_dif_contrasts(dif)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    # benjamini_hochberg import used above to keep the dependency explicit
    assert benjamini_hochberg([0.01, 0.5]).shape == (2,)


def test_plot_group_ability_distributions() -> None:
    n = 300
    groups = ["ref"] * (n // 2) + ["focal"] * (n // 2)
    df = simulate_rasch_data(n_persons=n, n_items=8, seed=3, groups=groups)
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    fig = plot_group_ability_distributions(model, groups)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
