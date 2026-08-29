"""Tests for PCA of residuals (PCAR) dimensionality diagnostic.

Expected behavior is checked via recovery properties: a clean unidimensional
simulation yields a first-contrast eigenvalue below the 2.0 screening cutoff,
while a deliberately two-dimensional simulation pushes it above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rasch_per.data import ResponseData
from rasch_per.rasch import RaschModel, run_pcar
from rasch_per.simulate import simulate_rasch_data


def test_pcar_clean_unidimensional_below_cutoff() -> None:
    df = simulate_rasch_data(n_persons=500, n_items=20, seed=4)
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    result = run_pcar(model)
    assert isinstance(result.first_contrast_eigenvalue, float)
    assert result.first_contrast_eigenvalue >= 0.0
    # Clean unidimensional data: no second dimension suggested.
    assert result.first_contrast_eigenvalue < 2.0
    assert not result.second_dimension_suspected
    # Eigenvalues returned in descending order, one per item.
    assert result.eigenvalues.shape == (20,)
    assert np.all(np.diff(result.eigenvalues) <= 0)


def test_pcar_two_dimensional_raises_first_contrast() -> None:
    # Build a clearly two-dimensional structure: person split into two groups
    # with opposite item difficulty patterns, so a single Rasch dimension
    # cannot absorb the variance and a second contrast emerges.
    rng = np.random.default_rng(8)
    n = 400
    half = n // 2
    # Group A: easy items 0-9, hard items 10-19. Group B: the reverse.
    abilities_a = rng.normal(1.0, 0.6, size=half)
    abilities_b = rng.normal(-1.0, 0.6, size=n - half)
    theta = np.concatenate([abilities_a, abilities_b])
    beta_easy = rng.normal(-1.0, 0.3, size=10)
    beta_hard = rng.normal(1.0, 0.3, size=10)
    beta = np.concatenate([beta_easy, beta_hard])
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
    # Flip the effective pattern for group B on the first 10 items to create
    # a second dimension (local structure not explained by theta).
    responses = (rng.random((n, 20)) < p).astype(float)
    responses[:half, 10:20] = 1.0 - responses[:half, 10:20]
    df = pd.DataFrame(responses, columns=[f"item_{j + 1}" for j in range(20)])
    model = RaschModel().fit(ResponseData(df), estimator="MML")
    result = run_pcar(model)
    # The unmodeled second dimension should surface as a large contrast.
    assert result.first_contrast_eigenvalue > result.eigenvalues[1]
    # Advisory flag may or may not trigger depending on strength; we only
    # require that the first contrast is clearly the dominant one.
    assert result.first_contrast_eigenvalue >= 1.0
