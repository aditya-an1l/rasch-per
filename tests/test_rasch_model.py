"""Tests for rasch_per.rasch.model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rasch_per.data import ResponseData
from rasch_per.rasch import RaschModel, rasch_logit, rasch_probability


class TestRaschFunctions:
    def test_equal_ability_difficulty_gives_half(self) -> None:
        assert float(rasch_probability(0.0, 0.0)) == 0.5

    def test_known_value(self) -> None:
        # expit(1) = 0.7310585786300049
        assert float(rasch_probability(1.0, 0.0)) == pytest.approx(0.7310585786)

    def test_logit_is_difference(self) -> None:
        assert float(rasch_logit(2.0, -1.0)) == pytest.approx(3.0)

    def test_broadcasting_shapes(self) -> None:
        theta = np.array([0.0, 1.0, 2.0])
        beta = np.array([-1.0, 0.0])
        out = rasch_probability(theta[:, None], beta[None, :])
        assert out.shape == (3, 2)
        assert np.all((out > 0) & (out < 1))

    def test_monotone_in_theta_and_beta(self) -> None:
        grid = np.linspace(-3, 3, 25)
        p = rasch_probability(grid, 0.0)
        assert np.all(np.diff(p) > 0)  # increasing in ability
        p = rasch_probability(0.0, grid)
        assert np.all(np.diff(p) < 0)  # decreasing in difficulty

    def test_extreme_values_do_not_overflow(self) -> None:
        p = rasch_probability(np.array([800.0]), np.array([-800.0]))
        assert np.isfinite(p).all() and float(p[0]) == pytest.approx(1.0)


@pytest.fixture
def fitted() -> RaschModel:
    from rasch_per.simulate import simulate_rasch_data

    df, params = simulate_rasch_data(n_persons=120, n_items=8, seed=7, return_parameters=True)
    rd = ResponseData(
        df.drop(columns="person_id").to_numpy(dtype=float),
        item_names=[f"item_{i}" for i in range(1, 9)],
        person_ids=df["person_id"].tolist(),
    )
    model = RaschModel().fit(rd, estimator="MML", max_iter=200)
    model._true = params  # noqa: SLF001 - test convenience only
    return model


class TestRaschModelFacade:
    def test_fit_returns_self(self, fitted: RaschModel) -> None:
        assert isinstance(fitted, RaschModel)

    def test_item_difficulties_series_indexed_by_items(self, fitted: RaschModel) -> None:
        s = fitted.item_difficulties
        assert isinstance(s, pd.Series)
        assert len(s) == 8
        assert s.index.tolist() == [f"item_{i}" for i in range(1, 9)]

    def test_person_abilities_indexed_by_persons(self, fitted: RaschModel) -> None:
        s = fitted.person_abilities
        assert len(s) == 120
        assert s.index[0] == "p1"

    def test_standard_errors_keys_and_lengths(self, fitted: RaschModel) -> None:
        se = fitted.standard_errors
        assert set(se) == {"item_difficulty", "person_ability"}
        assert np.all(se["item_difficulty"] > 0)
        assert np.all(se["person_ability"] > 0)

    def test_unfitted_access_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="Call .fit()"):
            _ = RaschModel().item_difficulties

    def test_unknown_estimator_raises(self) -> None:
        from rasch_per.simulate import simulate_rasch_data

        df = simulate_rasch_data(10, 4, seed=1)
        with pytest.raises(ValueError, match="MML"):
            RaschModel().fit(ResponseData(df.drop(columns=["item_1"])), estimator="BOGUS")

    def test_fit_statistics_not_until_phase_3(self, fitted: RaschModel) -> None:
        with pytest.raises(NotImplementedError, match="Phase 3"):
            fitted.fit_statistics()
