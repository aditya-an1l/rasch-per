"""Tests for rasch_per.reliability against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from rasch_per.reliability import cronbach_alpha, ferguson_delta, mcdonald_omega

TOY = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)


class TestCronbachAlpha:
    def test_toy_reference_value(self) -> None:
        assert cronbach_alpha(TOY) == pytest.approx(27 / 34)

    def test_perfectly_consistent_items_give_one(self) -> None:
        col = np.array([[1.0], [1.0], [0.0], [0.0]])
        x = np.hstack([col] * 3)
        assert cronbach_alpha(x) == pytest.approx(1.0)

    def test_two_items_minimum(self) -> None:
        x = np.array([[1, 1], [1, 0], [0, 0], [0, 1]], dtype=float)
        assert -1.0 <= cronbach_alpha(x) <= 1.0

    def test_single_item_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 items"):
            cronbach_alpha(np.array([[1.0], [0.0]]))

    def test_nan_rows_dropped_listwise(self) -> None:
        with_nan = TOY.copy()
        with_nan[0, 0] = np.nan
        # Listwise deletion keeps rows p2..p5 of the original toy matrix.
        assert cronbach_alpha(with_nan) == pytest.approx(cronbach_alpha(TOY[1:]))


class TestMcDonaldOmega:
    def test_in_unit_interval(self) -> None:
        assert 0.0 <= mcdonald_omega(TOY) <= 1.0

    def test_strong_single_factor_gives_high_omega(self) -> None:
        # Items designed to be nearly parallel: high omega expected.
        rng = np.random.default_rng(11)
        latent = rng.normal(size=200)
        threshold = 0.0
        items = (latent[:, None] + rng.normal(scale=0.2, size=(200, 4))) > threshold
        assert mcdonald_omega(items.astype(float)) > 0.85

    def test_uncorrelated_items_give_low_omega(self) -> None:
        rng = np.random.default_rng(3)
        noise = rng.random((500, 4)) > 0.7
        assert mcdonald_omega(noise.astype(float)) < 0.5

    def test_fewer_than_three_items_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3 items"):
            mcdonald_omega(TOY[:, :2])

    def test_constant_items_return_nan(self) -> None:
        constant = np.ones((10, 3))
        assert np.isnan(mcdonald_omega(constant))


class TestFergusonDelta:
    def test_toy_reference_value(self) -> None:
        # Totals [3,2,1,0,0]: f^2 sum = 16+1+1+1... wait: f_3=1,f_2=1,f_1=1,
        # f_0=2 -> sum f^2 = 1+1+1+4 = 7; delta = 18/18.75 = 0.96.
        assert ferguson_delta(TOY) == pytest.approx(0.96)

    def test_uniform_score_distribution_gives_one(self) -> None:
        # One person per possible total score on k=3 items.
        x = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float)
        assert ferguson_delta(x) == pytest.approx(1.0)

    def test_all_identical_totals_give_zero(self) -> None:
        x = np.zeros((6, 3))
        assert ferguson_delta(x) == pytest.approx(0.0)

    def test_docstring_caution_note_present(self) -> None:
        doc = ferguson_delta.__doc__ or ""
        assert "population-dependent" in doc
