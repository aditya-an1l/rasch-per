"""Tests for item/person fit statistics (infit, outfit, Q3).

Reference values follow the reference-value-testing skill: expected values
are recomputed independently from first principles in numpy (a second method),
not transcribed by hand, and misfit recovery is checked via the simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from rasch_per.data import ResponseData
from rasch_per.rasch import (
    RaschModel,
    compute_q3_matrix,
    flag_misfitting_items,
    flag_q3_violations,
    infit_outfit,
)
from rasch_per.rasch.fit import FIT_PRESETS, standardized_residuals


def _fit_jml(df: pd.DataFrame) -> RaschModel:
    return RaschModel().fit(ResponseData(df), estimator="JML")


def _independent_infit_outfit(model: RaschModel) -> tuple[np.ndarray, np.ndarray]:
    """Recompute infit/outfit from first principles (second method)."""
    matrix = model.responses
    betas = np.asarray(model.item_difficulties, dtype=float)
    thetas = np.asarray(model.person_abilities, dtype=float)
    p = expit(thetas[:, None] - betas[None, :])
    observed = np.isfinite(matrix)
    z = np.where(observed, (matrix - p) / np.sqrt(p * (1.0 - p)), np.nan)
    z2 = np.where(observed, z**2, np.nan)
    info = np.where(observed, p * (1.0 - p), np.nan)
    outfit = np.nanmean(z2, axis=0)
    infit = np.nansum(z2 * info, axis=0) / np.nansum(info, axis=0)
    return infit, outfit


def test_infit_outfit_matches_independent_recompute() -> None:
    rng = np.random.default_rng(123)
    matrix = rng.integers(0, 2, size=(40, 8)).astype(float)
    model = _fit_jml(pd.DataFrame(matrix, columns=[f"i{j}" for j in range(8)]))
    fit = infit_outfit(model)
    infit, outfit = _independent_infit_outfit(model)
    assert list(fit.index) == [f"i{j}" for j in range(8)]
    np.testing.assert_allclose(fit["infit"].to_numpy(), infit, atol=1e-10)
    np.testing.assert_allclose(fit["outfit"].to_numpy(), outfit, atol=1e-10)


def test_infit_outfit_near_one_for_clean_sim() -> None:
    from rasch_per.simulate import simulate_rasch_data

    df = simulate_rasch_data(n_persons=600, n_items=20, seed=7)
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    fit = infit_outfit(model)
    # Clean unidimensional data: statistics center on 1.0.
    assert fit["infit"].mean() == pytest.approx(1.0, abs=0.15)
    assert fit["outfit"].mean() == pytest.approx(1.0, abs=0.15)
    assert np.all(np.isfinite(fit.to_numpy()))


def test_misfit_items_show_elevated_outfit() -> None:
    from rasch_per.simulate import simulate_rasch_data

    df = simulate_rasch_data(
        n_persons=500, n_items=15, seed=11, misfit_items=[0, 1], misfit_prob=0.3
    )
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    fit = infit_outfit(model)
    misfit = fit.loc[[f"item_{1}", f"item_{2}"]].mean()
    clean = fit.loc[[c for c in fit.index if c not in (f"item_{1}", f"item_{2}")]].mean()
    # Injected misfit should inflate fit statistics relative to clean items.
    assert misfit["outfit"] > clean["outfit"]
    assert misfit["infit"] >= clean["infit"]


def test_flag_misfitting_items_structure_and_bounds() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.integers(0, 2, size=(50, 6)).astype(float)
    model = _fit_jml(pd.DataFrame(matrix, columns=[f"q{j}" for j in range(6)]))
    flagged = flag_misfitting_items(model, "high_stakes")
    assert list(flagged.columns) == [
        "infit",
        "outfit",
        "infit_misfit",
        "outfit_misfit",
    ]
    lower, upper = FIT_PRESETS["high_stakes"]
    expected_infit = (flagged["infit"] < lower) | (flagged["infit"] > upper)
    expected_outfit = (flagged["outfit"] < lower) | (flagged["outfit"] > upper)
    np.testing.assert_array_equal(flagged["infit_misfit"].to_numpy(), expected_infit.to_numpy())
    np.testing.assert_array_equal(flagged["outfit_misfit"].to_numpy(), expected_outfit.to_numpy())


def test_flag_bounds_custom_and_presets() -> None:
    rng = np.random.default_rng(9)
    matrix = rng.integers(0, 2, size=(40, 5)).astype(float)
    model = _fit_jml(pd.DataFrame(matrix, columns=[f"q{j}" for j in range(5)]))
    # Custom tuple straddling 1.0 is accepted.
    flag_misfitting_items(model, (0.5, 1.5))
    # Non-straddling tuple raises.
    with pytest.raises(ValueError):
        flag_misfitting_items(model, (0.5, 0.8))
    # Unknown preset raises.
    with pytest.raises(ValueError):
        flag_misfitting_items(model, "nonsense")


def test_standardized_residuals_shape_and_missing() -> None:
    df = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, np.nan]})
    model = _fit_jml(df)
    z = standardized_residuals(model)
    assert z.shape == (3, 2)
    assert np.isnan(z[2, 1])  # missing response -> NaN residual


def test_q3_clean_sim_low_correlation() -> None:
    from rasch_per.simulate import simulate_rasch_data

    df = simulate_rasch_data(n_persons=500, n_items=15, seed=3)
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    q3 = compute_q3_matrix(model)
    assert q3.shape == (15, 15)
    np.testing.assert_array_equal(q3.to_numpy(), q3.to_numpy().T)  # symmetric
    off_diag = q3.to_numpy()[~np.eye(15, dtype=bool)]
    # Unidimensional clean data: residual correlations stay small.
    assert np.abs(off_diag).mean() < 0.15


def test_q3_detects_local_dependence() -> None:
    # Two identical item columns are perfectly locally dependent.
    df = pd.DataFrame(
        {
            "a": [1, 0, 1, 0, 1, 0],
            "b": [1, 0, 1, 0, 1, 0],  # exact duplicate of "a"
            "c": [0, 1, 0, 1, 0, 1],
        }
    )
    model = _fit_jml(df)
    q3 = compute_q3_matrix(model)
    violations = flag_q3_violations(q3, threshold=0.5)
    pairs = set(zip(violations["item_a"], violations["item_b"], strict=False))
    assert ("a", "b") in pairs
    # Each violation reports the absolute correlation.
    assert (violations["q3"].abs() > 0.5).all()
