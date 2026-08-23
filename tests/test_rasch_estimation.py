"""Recovery and unit tests for rasch_per.rasch.estimation.

Ground truth comes from simulate_rasch_data: we simulate from known
theta/beta, refit with each estimator, and assert parameter recovery.
"""

from __future__ import annotations

import numpy as np
import pytest

from rasch_per.rasch.estimation import fit_jml, fit_mml
from rasch_per.simulate import simulate_rasch_data


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def truth():
    df, params = simulate_rasch_data(n_persons=500, n_items=20, seed=42, return_parameters=True)
    matrix = df.drop(columns="person_id").to_numpy(dtype=float)
    return matrix, params


class TestJMLRecovery:
    def test_beta_recovery_above_threshold(self, truth) -> None:
        matrix, params = truth
        result = fit_jml(matrix, max_iter=300)
        r = pearson_r(result.betas, params["beta"])
        assert r > 0.95, f"JML beta recovery r={r:.3f}"

    def test_theta_recovery_attenuated_but_strong(self, truth) -> None:
        # With 20 dichotomous items, person-level attenuation caps the
        # attainable correlation near sqrt(reliability); > 0.8 is strong.
        matrix, params = truth
        result = fit_jml(matrix, max_iter=300)
        r = pearson_r(result.thetas, params["theta"])
        assert r > 0.80, f"JML theta recovery r={r:.3f}"

    def test_identifiability_constraint_mean_beta_zero(self, truth) -> None:
        matrix, _ = truth
        assert fit_jml(matrix).betas.mean() == pytest.approx(0.0, abs=1e-10)

    def test_se_magnitudes_sane(self, truth) -> None:
        matrix, _ = truth
        result = fit_jml(matrix, max_iter=200)
        assert np.all(np.isfinite(result.se_beta)) and np.all(result.se_beta > 0)
        rmse = np.sqrt(np.mean((result.betas - truth[1]["beta"]) ** 2))
        mean_se = np.mean(result.se_beta)
        assert rmse / mean_se < 2.5  # SEs within the right order of magnitude

    def test_extreme_scorer_gets_finite_estimate(self) -> None:
        df = simulate_rasch_data(n_persons=100, n_items=10, seed=5)
        matrix = df.drop(columns="person_id").to_numpy(dtype=float).copy()
        matrix[0] = 1.0  # force an all-correct person
        with pytest.warns(UserWarning, match="Extreme scores"):
            result = fit_jml(matrix)
        assert np.isfinite(result.thetas[0])

    def test_unadministered_item_raises(self) -> None:
        matrix = np.full((20, 4), np.nan)
        matrix[:, 0] = 1.0
        with pytest.raises(ValueError, match="never administered"):
            fit_jml(matrix)


class TestMMLRecovery:
    def test_beta_recovery_above_threshold(self, truth) -> None:
        matrix, params = truth
        result = fit_mml(matrix, max_iter=300)
        r = pearson_r(result.betas, params["beta"])
        assert r > 0.95, f"MML beta recovery r={r:.3f}"

    def test_theta_eap_recovery(self, truth) -> None:
        matrix, params = truth
        result = fit_mml(matrix, max_iter=300)
        r = pearson_r(result.thetas, params["theta"])
        # EAP estimates are shrunk toward 0 by the N(0,1) prior; 0.75+ is
        # expected at this test length.
        assert r > 0.70, f"MML theta recovery r={r:.3f}"

    def test_convergence_flag_true(self, truth) -> None:
        matrix, _ = truth
        assert fit_mml(matrix, max_iter=300).converged

    def test_se_positive_and_finite(self, truth) -> None:
        matrix, _ = truth
        result = fit_mml(matrix)
        assert np.all(np.isfinite(result.se_beta))
        assert np.all(result.se_beta > 0)

    def test_missing_cells_handled(self, truth) -> None:
        matrix, params = truth
        holed = matrix.copy()
        rng = np.random.default_rng(0)
        hole_idx = (rng.integers(0, holed.shape[0], 200), rng.integers(0, holed.shape[1], 200))
        holed[hole_idx] = np.nan
        result = fit_mml(holed, max_iter=300)
        assert pearson_r(result.betas, params["beta"]) > 0.9

    def test_matches_jml_closely_on_clean_data(self, truth) -> None:
        # Both estimators target the same true parameters; their beta
        # vectors should agree far better than either agrees with chance.
        matrix, _ = truth
        mml = fit_mml(matrix, max_iter=300).betas
        jml = fit_jml(matrix, max_iter=300).betas
        assert pearson_r(mml, jml) > 0.98


class TestEstimatorContract:
    def test_result_is_frozen_dataclass(self, truth) -> None:
        matrix, _ = truth
        result = fit_mml(matrix[:50], max_iter=50)
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            result.betas = np.zeros(20)  # type: ignore[misc]

    def test_shapes_match_input(self, truth) -> None:
        matrix, _ = truth
        for est in (fit_jml(matrix), fit_mml(matrix)):
            assert est.betas.shape == (20,)
            assert est.thetas.shape == (500,)
