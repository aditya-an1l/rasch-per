"""Tests for rasch_per.simulate."""

from __future__ import annotations

import numpy as np
import pytest

from rasch_per.simulate import simulate_rasch_data


class TestBasicGeneration:
    def test_shape_and_columns(self) -> None:
        df = simulate_rasch_data(n_persons=30, n_items=6, seed=0)
        assert df.shape == (30, 7)
        assert list(df.columns) == ["person_id"] + [f"item_{i}" for i in range(1, 7)]

    def test_values_dichotomous(self) -> None:
        df = simulate_rasch_data(50, 5, seed=1)
        assert set(np.unique(df.iloc[:, 1:].to_numpy())) <= {0, 1}

    def test_seed_reproducibility(self) -> None:
        a = simulate_rasch_data(40, 8, seed=123)
        b = simulate_rasch_data(40, 8, seed=123)
        assert a.equals(b)

    def test_return_parameters(self) -> None:
        df, params = simulate_rasch_data(25, 4, seed=2, return_parameters=True)
        assert params["theta"].shape == (25,)
        assert params["beta"].shape == (4,)

    def test_custom_distributions_shift_difficulty(self) -> None:
        _, easy = simulate_rasch_data(
            400, 10, beta_dist=(-2.0, 0.5), seed=3, return_parameters=True
        )
        _, hard = simulate_rasch_data(400, 10, beta_dist=(2.0, 0.5), seed=3, return_parameters=True)
        assert easy["beta"].mean() < hard["beta"].mean()

    def test_invalid_sizes_raise(self) -> None:
        with pytest.raises(ValueError):
            simulate_rasch_data(n_persons=0)


class TestDIFInjection:
    def test_requires_groups(self) -> None:
        with pytest.raises(ValueError, match="groups"):
            simulate_rasch_data(dif_effects={0: 1.0})

    def test_focal_group_find_item_harder(self) -> None:
        n = 600
        groups = np.array(["ref", "focal"] * (n // 2))
        df, params = simulate_rasch_data(
            n_persons=n,
            n_items=10,
            seed=9,
            theta_dist=(0.0, 1.0),
            groups=groups,
            focal_label="focal",
            dif_effects={2: 2.0},
            return_parameters=True,
        )
        item3 = df["item_3"].to_numpy()
        ref_correct = item3[groups == "ref"].mean()
        foc_correct = item3[groups == "focal"].mean()
        # A +2 logit shift should make the focal group noticeably less correct.
        assert ref_correct - foc_correct > 0.15
        # Ground-truth base beta is unchanged by the injection.
        assert len(params["beta"]) == 10

    def test_non_target_items_unaffected_on_average(self) -> None:
        n = 600
        groups = np.array(["ref", "focal"] * (n // 2))
        df = simulate_rasch_data(
            n_persons=n,
            n_items=8,
            seed=11,
            groups=groups,
            dif_effects={0: 3.0},
        )
        for col in ("item_5", "item_6"):
            vals = df[col].to_numpy()
            assert abs(vals[groups == "ref"].mean() - vals[groups == "focal"].mean()) < 0.15


class TestMisfitInjection:
    def test_misfit_flips_responses(self) -> None:
        clean = simulate_rasch_data(300, 12, seed=21)
        dirty = simulate_rasch_data(300, 12, seed=21, misfit_items=[4], misfit_prob=0.5)
        flipped = (clean["item_5"] != dirty["item_5"]).sum()
        assert flipped > 50  # roughly half of 300 with prob .5

    def test_other_items_identical_without_misfit(self) -> None:
        clean = simulate_rasch_data(100, 12, seed=21)
        dirty = simulate_rasch_data(100, 12, seed=21, misfit_items=[4], misfit_prob=0.5)
        assert clean["item_7"].equals(dirty["item_7"])
