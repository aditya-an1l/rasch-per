"""Tests for information functions and Rasch reliability.

Expected values are derived from the analytic formulas themselves (p(1-p),
1/sqrt(TIF)) and from recovery properties on simulated data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rasch_per.data import ResponseData
from rasch_per.rasch import (
    RaschModel,
    item_information,
    person_separation_reliability,
    sem,
)
from rasch_per.rasch import (
    test_information as tif,
)
from rasch_per.simulate import simulate_rasch_data


def test_item_information_at_beta_is_quarter() -> None:
    # At theta == beta, p = 0.5, so I = 0.25.
    assert float(item_information(0.0, 0.0)) == pytest.approx(0.25)
    assert float(item_information(1.5, 1.5)) == pytest.approx(0.25)


def test_item_information_decreases_away_from_beta() -> None:
    info_center = float(item_information(0.0, 0.0))
    info_away = float(item_information(0.0, 2.0))
    assert info_center > info_away > 0.0


def test_test_information_is_sum_of_item_information() -> None:
    betas = np.array([0.0, 0.5, -0.3])
    theta = np.array([-1.0, 0.0, 1.0])
    result = tif(betas, theta)
    expected = np.array([sum(item_information(b, t) for b in betas) for t in theta])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_sem_is_inverse_sqrt_tif() -> None:
    betas = np.array([-0.2, 0.4, 0.9])
    theta = 0.0
    expected = 1.0 / np.sqrt(tif(betas, theta))
    assert float(sem(betas, theta)) == pytest.approx(float(expected))


def test_information_scalar_input_returns_scalar() -> None:
    out = tif(np.array([0.0, 1.0]), 0.5)
    assert np.isscalar(out) or out.shape == ()  # type: ignore[attr-defined]
    out2 = sem(np.array([0.0, 1.0]), 0.5)
    assert np.isscalar(out2) or out2.shape == ()  # type: ignore[attr-defined]


def test_person_separation_reliability_clean_sim_high() -> None:
    df = simulate_rasch_data(n_persons=500, n_items=25, seed=21)
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    rel = person_separation_reliability(model)
    assert 0.0 <= rel <= 1.0
    # Good separation with many items and persons (25 items yields ~0.7).
    assert rel > 0.6


def test_person_separation_reliability_tiny_sample_nan() -> None:
    df = pd.DataFrame({"a": [1, 0], "b": [0, 1]})
    model = RaschModel().fit(ResponseData(df), estimator="JML")
    assert np.isnan(person_separation_reliability(model))
