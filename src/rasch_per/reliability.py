"""Additional reliability coefficients.

Cronbach's alpha, McDonald's omega (via a single-factor model on the Pearson
correlation matrix using an eigenvalue-based approximation - documented), and
Ferguson's delta:

    delta = (N^2 - sum(f_x^2)) / (N^2 - N^2 / (k + 1))

where N = number of respondents, f_x = frequency of each total score x,
k = number of items. Ferguson's delta is population-dependent and should be
interpreted with caution.

Spec reference: sections 6.1-6.2 of the project build spec.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cronbach_alpha", "mcdonald_omega", "ferguson_delta"]


def _complete_rows(x: np.ndarray) -> np.ndarray:
    """Listwise-complete a response matrix, raising if too little remains."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2-D person x item matrix")
    complete = arr[~np.isnan(arr).any(axis=1)]
    if complete.shape[0] < 2:
        raise ValueError(
            "Need at least 2 persons with complete responses on all items "
            "for reliability estimation"
        )
    return complete


def cronbach_alpha(x: np.ndarray) -> float:
    """Cronbach's alpha internal-consistency reliability.

    Computed with listwise deletion of rows containing NaN and sample
    variances (ddof=1):

        alpha = k / (k - 1) * (1 - sum_i Var(item_i) / Var(total))

    Parameters
    ----------
    x : array-like, shape (n_persons, n_items)
        Dichotomous (or continuous) item-score matrix.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If fewer than 2 items or 2 complete respondents.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> round(cronbach_alpha(x), 4)
    0.7941
    """
    z = _complete_rows(x)
    k = z.shape[1]
    if k < 2:
        raise ValueError("Cronbach's alpha needs at least 2 items")
    total_var = z.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    item_var_sum = z.var(axis=0, ddof=1).sum()
    return float(k / (k - 1) * (1.0 - item_var_sum / total_var))


def mcdonald_omega(x: np.ndarray) -> float:
    """McDonald's omega total reliability via an eigenvalue approximation.

    A single common factor is extracted from the Pearson correlation matrix
    by taking the first eigenvector scaled to correlation loadings
    (lambda_i = v_i * sqrt(eigenvalue)). Model-implied uniquenesses are
    u_i = 1 - lambda_i^2 and

        omega = (sum lambda)^2 / ((sum lambda)^2 + sum u)

    This is a lightweight approximation to a full single-factor SEM solution;
    it matches omega exactly only when the first factor explains the observed
    correlations perfectly. It is documented here as such and is sufficient
    for the reporting conventions of education-research instrument validation.

    Parameters
    ----------
    x : array-like, shape (n_persons, n_items)
        Dichotomous item-score matrix (listwise deletion of NaN rows).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If fewer than 3 items (a single-factor identification minimum here)
        or too few complete respondents.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> 0.0 <= mcdonald_omega(x) <= 1.0
    True
    """
    z = _complete_rows(x)
    k = z.shape[1]
    if k < 3:
        raise ValueError("McDonald's omega needs at least 3 items")
    if np.any(z.std(axis=0) == 0):
        return float("nan")  # constant items make correlations undefined
    corr = np.corrcoef(z, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return float("nan")
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    top = int(np.argmax(eigenvalues))
    loading = eigenvectors[:, top] * np.sqrt(max(eigenvalues[top], 0.0))
    if loading.sum() < 0:  # eigenvector sign is arbitrary; orient positively
        loading = -loading
    loading_sq = np.clip(loading**2, 0.0, 1.0)
    uniqueness = np.clip(1.0 - loading_sq, 0.0, None)
    true_var = float(loading.sum() ** 2)
    denom = true_var + float(uniqueness.sum())
    if denom == 0:
        return float("nan")
    return float(true_var / denom)


def ferguson_delta(x: np.ndarray) -> float:
    """Ferguson's delta discriminatory power index.

        delta = (N^2 - sum_x f_x^2) / (N^2 - N^2 / (k + 1))

    where N is the number of respondents, f_x the frequency of each possible
    total score x in {0, ..., k} and k the number of items. The index
    compares the observed spread of total scores with a uniform distribution
    over all k + 1 scores; values above ~0.9 indicate good discriminatory
    power for norm-referenced tests.

    Ferguson's delta is population-dependent: it reflects how spread out this
    particular sample's scores are, not an intrinsic property of the items.
    Interpret it with caution and never compare across samples with different
    ability distributions.

    Parameters
    ----------
    x : array-like, shape (n_persons, n_items)
        Dichotomous item-score matrix (NaN cells are treated as 0 in total
        scores, matching the convention of scoring unadministered items as
        incorrect; filter such persons beforehand with
        ``ResponseData.filter_min_response_rate``).

    Returns
    -------
    float

    References
    ----------
    Ferguson, G. A. (1949). On the theory of test discrimination.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> round(ferguson_delta(x), 2)
    0.96
    """
    arr = np.asarray(x, dtype=float)
    n_persons, n_items = arr.shape
    totals = np.nansum(arr, axis=1)
    freq = np.bincount(totals.astype(int), minlength=n_items + 1)[: n_items + 1]
    denominator = n_persons**2 * n_items / (n_items + 1)
    if denominator == 0:
        return float("nan")
    numerator = n_persons**2 - int((freq**2).sum())
    return float(numerator / denominator)
