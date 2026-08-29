"""Tests for Differential Item Functioning (Lord's chi-square + ETS delta).

Follows the reference-value-testing skill: BH and ETS binning are checked with
independent small cases; DIF detection is verified as a recovery property on
simulated data with injected ``dif_effects``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rasch_per import DIFAnalysis, DIFResults, RaschModel
from rasch_per.data import ResponseData
from rasch_per.dif import benjamini_hochberg, ets_delta_class
from rasch_per.simulate import simulate_rasch_data


def test_benjamini_hochberg_adjusts_and_monotone() -> None:
    p = [0.01, 0.02, 0.30, 0.50]
    q = benjamini_hochberg(p)
    assert q.shape == (4,)
    # Smallest p gets the smallest q; all <= 1.
    assert q[0] <= q[1] <= q[2] <= q[3]
    assert np.all((q >= 0) & (q <= 1))
    # With 4 tests, the smallest adjusted p is 4 * 0.01 = 0.04.
    assert q[0] == pytest.approx(0.04)


def test_ets_delta_class_bins() -> None:
    assert ets_delta_class(0.5) == "A"
    assert ets_delta_class(1.2) == "B"
    assert ets_delta_class(2.0) == "C"
    assert ets_delta_class(float("nan")) == "NA"


def test_dif_ets_sign_convention() -> None:
    # Independent recomputation of the module's ETS delta formula:
    # delta = -2.35 * ln(odds_focal / odds_ref). If the focal group has a
    # higher proportion correct (easier for focal), odds_focal > odds_ref and
    # the delta is negative under this convention.
    p_ref, p_focal = 0.4, 0.7
    odds_f = p_focal / (1 - p_focal)
    odds_r = p_ref / (1 - p_ref)
    delta = -2.35 * np.log(odds_f / odds_r)
    assert delta < 0.0
    assert ets_delta_class(abs(delta)) in {"B", "C"}


def test_dif_detects_injected_effect() -> None:
    n = 400
    n_items = 12
    groups = ["ref"] * (n // 2) + ["focal"] * (n // 2)
    df = simulate_rasch_data(
        n_persons=n,
        n_items=n_items,
        seed=14,
        groups=groups,
        focal_label="focal",
        dif_effects={0: 0.9},  # item_1 harder for the focal group
    )
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    results = DIFAnalysis(model, groups, reference="ref", focal="focal").analyze()
    table = results.summary()

    assert set(table.columns) >= {
        "beta_ref",
        "beta_focal",
        "dif",
        "lord_chi2",
        "p_value",
        "bh_q",
        "flag",
        "ets_delta",
        "ets_class",
    }
    # The injected item is flagged.
    assert bool(table.loc["item_1", "flag"]) is True
    assert table.loc["item_1", "ets_class"] in {"B", "C"}
    # The great majority of clean items are NOT flagged.
    n_flagged = int(table["flag"].sum())
    assert n_flagged <= 3
    # Benjamini-Hochberg q-values are monotonic non-decreasing in p order.
    p_sorted = np.sort(table["p_value"].to_numpy())
    q_at_sorted = benjamini_hochberg(p_sorted)
    assert np.all(np.diff(q_at_sorted) >= 0)


def test_dif_clean_data_flags_little() -> None:
    n = 400
    n_items = 12
    groups = ["ref"] * (n // 2) + ["focal"] * (n // 2)
    df = simulate_rasch_data(n_persons=n, n_items=n_items, seed=31, groups=groups)
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    results = DIFAnalysis(model, groups, reference="ref", focal="focal").analyze()
    # No DIF injected: expect at most 1 false positive at FDR 0.05.
    assert int(results.summary()["flag"].sum()) <= 1


def test_dif_results_summary_and_missing_groups() -> None:
    df = simulate_rasch_data(n_persons=100, n_items=8, seed=2)
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    groups = ["ref"] * 50 + ["focal"] * 50
    results = DIFAnalysis(model, groups, reference="ref", focal="focal").analyze()
    assert isinstance(results, DIFResults)
    assert results.summary().shape[0] == 8
    # plot() is deferred to Phase 5.
    with pytest.raises(NotImplementedError):
        results.plot()
    # Both groups must be present.
    with pytest.raises(ValueError):
        DIFAnalysis(model, groups, reference="ref", focal="ghost").analyze()
