"""Tests for rasch_per.ctt against hand-computed reference values.

The toy dataset (5 persons x 3 items, tests/data/synthetic_small.csv):

        q1  q2  q3
    p1   1   1   1
    p2   1   0   1
    p3   1   0   0
    p4   0   0   0
    p5   0   0   0

Hand computations (see docs in tests/data/reference_values.json):
- difficulties: 3/5, 1/5, 2/5
- rest scores per item (total minus that item):
    q1 rest: [2, 1, 0, 0, 0]
    q2 rest: [2, 2, 1, 0, 0]
    q3 rest: [2, 1, 1, 0, 0]
- discriminations (item vs REST score, Pearson r):
    q1: 1.2 / sqrt(1.2 * 3.2) = 0.6123724...
    q2: 1.0 / sqrt(0.8 * 4.0) = 0.5590170...
    q3: 1.4 / sqrt(1.2 * 2.8) = 0.7637626...
- Cronbach's alpha (ddof=1): 3/2 * (1 - 0.8/1.7) = 27/34 = 0.7941176...
- Ferguson's delta: (25 - (4+1+1+1)) / (25 - 6.25) = 18/18.75 = 0.96
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from rasch_per.ctt import CTTAnalysis, CTTResults
from rasch_per.data import ResponseData


@pytest.fixture
def reference() -> dict:
    with open("tests/data/reference_values.json") as fh:
        return json.load(fh)


@pytest.fixture
def results(reference: dict) -> CTTResults:
    data = ResponseData(np.array(reference["toy_matrix"], dtype=float))
    return CTTAnalysis(data, n_boot=500, seed=42).run()


class TestItemStatistics:
    def test_difficulty_matches_reference(self, results: CTTResults, reference: dict) -> None:
        np.testing.assert_allclose(results.difficulty, reference["item_difficulties"], atol=1e-12)

    def test_discrimination_matches_reference(self, results: CTTResults, reference: dict) -> None:
        np.testing.assert_allclose(
            results.discrimination, reference["item_discriminations"], atol=1e-6
        )

    def test_bootstrap_se_finite_and_nonnegative(self, results: CTTResults) -> None:
        for se in (results.difficulty_se, results.discrimination_se):
            assert np.all(np.isfinite(se))
            assert np.all(se >= 0)

    def test_discrimination_uses_rest_score_not_total(self) -> None:
        # If discrimination used the full total score, values would be
        # inflated; check a case where the difference is detectable.
        matrix = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]], dtype=float)
        rd = ResponseData(matrix)
        out = CTTAnalysis(rd, n_boot=10, seed=0).run()
        rest_based = _hand_rest_discrimination(matrix, 0)
        assert out.discrimination[0] == pytest.approx(rest_based, abs=1e-12)


def _hand_rest_discrimination(matrix: np.ndarray, item: int) -> float:
    """Reference point-biserial of an item vs its rest score."""
    rest = matrix.sum(axis=1) - matrix[:, item]
    return float(np.corrcoef(matrix[:, item], rest)[0, 1])


class TestReliabilityObject:
    def test_alpha_matches_reference(self, results: CTTResults, reference: dict) -> None:
        assert results.reliability.cronbach_alpha == pytest.approx(reference["cronbach_alpha"])

    def test_ferguson_delta_matches_reference(self, results: CTTResults, reference: dict) -> None:
        assert results.reliability.ferguson_delta == pytest.approx(reference["ferguson_delta"])

    def test_omega_in_unit_interval(self, results: CTTResults) -> None:
        omega = results.reliability.mcdonald_omega
        assert 0.0 <= omega <= 1.0


class TestSummary:
    def test_summary_columns_and_rows(self, results: CTTResults) -> None:
        summary = results.summary()
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == [
            "item",
            "difficulty",
            "difficulty_se",
            "discrimination",
            "discrimination_se",
        ]
        assert len(summary) == 3
        assert summary["item"].tolist() == ["item_1", "item_2", "item_3"]

    def test_item_names_preserved_from_dataframe(self) -> None:
        df = pd.DataFrame({"q1": [1, 0], "q2": [0, 1]})
        out = CTTAnalysis(ResponseData(df), n_boot=10, seed=1).run()
        assert out.summary()["item"].tolist() == ["q1", "q2"]


class TestMissingHandling:
    def test_difficulty_ignores_nan(self) -> None:
        matrix = np.array([[1.0, 1.0], [np.nan, 0.0], [1.0, 0.0]])
        difficulty = CTTAnalysis(ResponseData(matrix), n_boot=20, seed=0).run().difficulty
        np.testing.assert_allclose(difficulty, [1.0, 1 / 3])

    def test_nan_free_analysis_runs_with_small_boot(self) -> None:
        matrix = np.eye(4, dtype=float)
        out = CTTAnalysis(ResponseData(matrix), n_boot=50, seed=7).run()
        assert len(out.summary()) == 4


class TestDataclassContract:
    def test_n_boot_recorded(self, results: CTTResults) -> None:
        assert results.n_boot == 500

    def test_reliability_is_frozen_dataclass(self, results: CTTResults) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            results.reliability.cronbach_alpha = 0.5  # type: ignore[misc]
