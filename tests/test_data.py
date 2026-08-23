"""Tests for rasch_per.data.ResponseData."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rasch_per.data import ResponseData


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "q1": [1, 0, 1, np.nan],
            "q2": [0, 1, 1, 1],
            "q3": [np.nan, 0, 1, 0],
        },
        index=["a", "b", "c", "d"],
    )


class TestValidation:
    def test_valid_values_pass(self, frame: pd.DataFrame) -> None:
        rd = ResponseData(frame)
        assert rd.n_persons == 4
        assert rd.n_items == 3

    def test_invalid_value_raises(self) -> None:
        df = pd.DataFrame({"q1": [1, 2]})
        with pytest.raises(ValueError, match=r"\{0, 1, NaN\}"):
            ResponseData(df)

    def test_string_value_raises_strict(self) -> None:
        df = pd.DataFrame({"q1": ["yes", "no"]})
        with pytest.raises(ValueError, match="strict=False"):
            ResponseData(df)

    def test_non_strict_coerces_strings(self) -> None:
        df = pd.DataFrame({"q1": ["yes", "No"], "q2": ["TRUE", "false"]})
        rd = ResponseData(df, strict=False)
        expected = np.array([[1.0, 1.0], [0.0, 0.0]])
        np.testing.assert_array_equal(rd.to_numpy(), expected)

    def test_non_strict_coerces_booleans_and_numbers(self) -> None:
        df = pd.DataFrame({"q1": [True, False], "q2": [7.0, 0.0]})
        rd = ResponseData(df, strict=False)
        np.testing.assert_array_equal(rd.to_numpy(), [[1.0, 1.0], [0.0, 0.0]])

    def test_empty_matrix_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ResponseData(pd.DataFrame({"q1": []}))


class TestProperties:
    def test_from_dataframe_with_person_id_column(self) -> None:
        df = pd.DataFrame({"person_id": ["u1", "u2"], "i1": [1, 0]})
        rd = ResponseData(df)
        assert rd.person_ids == ["u1", "u2"]
        assert rd.item_names == ["i1"]

    def test_from_numpy(self) -> None:
        arr = np.array([[1, 0], [0, 1]])
        rd = ResponseData(arr, item_names=["x", "y"])
        assert rd.item_names == ["x", "y"]
        np.testing.assert_array_equal(rd.to_numpy(), arr)

    def test_to_numpy_dtype_is_float(self, frame: pd.DataFrame) -> None:
        assert ResponseData(frame).to_numpy().dtype == float

    def test_repr_mentions_shape(self, frame: pd.DataFrame) -> None:
        text = repr(ResponseData(frame))
        assert "n_persons=4" in text and "n_items=3" in text


class TestMissingness:
    def test_missing_by_item_percentages(self, frame: pd.DataFrame) -> None:
        pct = ResponseData(frame).missing_by_item()
        assert pct["q1"] == pytest.approx(25.0)
        assert pct["q3"] == pytest.approx(25.0)
        assert pct["q2"] == pytest.approx(0.0)

    def test_missing_by_person_percentages(self, frame: pd.DataFrame) -> None:
        pct = ResponseData(frame).missing_by_person()
        assert pct["a"] == pytest.approx(100 / 3)
        assert pct["b"] == pytest.approx(0.0)


class TestFilterMinResponseRate:
    def test_drops_below_threshold(self, frame: pd.DataFrame) -> None:
        filtered = ResponseData(frame).filter_min_response_rate(0.5)
        # person a answered 2/3 (>= .5 kept), person d answered 3/3.
        assert sorted(filtered.person_ids) == ["a", "b", "c", "d"]

    def test_high_threshold_drops_partial_responders(self, frame: pd.DataFrame) -> None:
        filtered = ResponseData(frame).filter_min_response_rate(0.99)
        # person d skipped q1, so only b and c answered everything.
        assert sorted(filtered.person_ids) == ["b", "c"]

    def test_original_unchanged(self, frame: pd.DataFrame) -> None:
        rd = ResponseData(frame)
        rd.filter_min_response_rate(0.99)
        assert rd.n_persons == 4

    def test_invalid_threshold_raises(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ResponseData(frame).filter_min_response_rate(1.5)


class TestFromCsv:
    def test_loads_toy_fixture(self) -> None:
        rd = ResponseData.from_csv("tests/data/synthetic_small.csv")
        assert rd.n_persons == 5 and rd.n_items == 3
        assert rd.person_ids == ["p1", "p2", "p3", "p4", "p5"]
