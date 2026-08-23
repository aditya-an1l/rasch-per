"""Classical Test Theory (CTT) statistics.

Per item: difficulty (proportion correct) and discrimination (point-biserial
correlation between the item score and the REST score, i.e. total excluding
that item), each with bootstrap standard errors.

Test level: Cronbach's alpha, McDonald's omega, Ferguson's delta, returned in
a :class:`CTTResults` dataclass with ``.summary()`` and a ``.reliability``
sub-object.

Spec reference: section 6.1 of the project build spec.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rasch_per.data import ResponseData
from rasch_per.reliability import cronbach_alpha, ferguson_delta, mcdonald_omega

__all__ = ["CTTAnalysis", "CTTResults"]


@dataclass(frozen=True)
class ReliabilityStats:
    """Test-level reliability coefficients.

    Attributes
    ----------
    cronbach_alpha : float
        Internal-consistency reliability (listwise complete cases).
    mcdonald_omega : float
        Single-factor reliability via eigenvalue approximation.
    ferguson_delta : float
        Discriminatory power index (population-dependent).
    """

    cronbach_alpha: float
    mcdonald_omega: float
    ferguson_delta: float


@dataclass(frozen=True)
class CTTResults:
    """Classical test theory results for one instrument.

    Attributes
    ----------
    item_names : list of str
    difficulty : ndarray, shape (n_items,)
        Proportion correct per item.
    difficulty_se : ndarray, shape (n_items,)
        Bootstrap standard errors of difficulty.
    discrimination : ndarray, shape (n_items,)
        Corrected item-total correlations (against the rest score).
    discrimination_se : ndarray, shape (n_items,)
        Bootstrap standard errors of discrimination.
    reliability : ReliabilityStats
        Test-level alpha / omega / Ferguson's delta.
    n_boot : int
        Number of bootstrap resamples used for the SEs.
    """

    item_names: list[str]
    difficulty: np.ndarray = field(repr=False)
    difficulty_se: np.ndarray = field(repr=False)
    discrimination: np.ndarray = field(repr=False)
    discrimination_se: np.ndarray = field(repr=False)
    reliability: ReliabilityStats
    n_boot: int = 1000

    def summary(self) -> pd.DataFrame:
        """Return a per-item summary DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns: item, difficulty, difficulty_se, discrimination,
            discrimination_se.
        """
        return pd.DataFrame(
            {
                "item": self.item_names,
                "difficulty": self.difficulty,
                "difficulty_se": self.difficulty_se,
                "discrimination": self.discrimination,
                "discrimination_se": self.discrimination_se,
            }
        )


def _item_difficulty(matrix: np.ndarray) -> np.ndarray:
    """Proportion correct per item, ignoring NaN."""
    return np.nanmean(matrix, axis=0)


def _rest_scores(matrix: np.ndarray) -> np.ndarray:
    """Total score excluding each item, per person (NaN-aware)."""
    totals = np.nansum(matrix, axis=1)
    return totals[:, None] - matrix


def _discrimination(matrix: np.ndarray) -> np.ndarray:
    """Corrected item-total correlation per item (item vs REST score).

    Uses only persons who answered the item; NaN handling follows the
    pairwise-complete convention. Returns NaN where a correlation is
    undefined (e.g. zero variance).
    """
    rests = _rest_scores(matrix)
    k = matrix.shape[1]
    out = np.full(k, np.nan)
    for j in range(k):
        col = matrix[:, j]
        mask = ~np.isnan(col) & ~np.isnan(rests[:, j])
        if mask.sum() < 3:
            continue
        x, y = col[mask], rests[mask, j]
        sx, sy = x.std(), y.std()
        if sx == 0 or sy == 0:
            continue
        out[j] = float(np.corrcoef(x, y)[0, 1])
    return out


def _bootstrap_se(
    matrix: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap SE of a per-item statistic over person resamples.

    Degenerate resamples (e.g. zero variance) yield NaN values that are
    ignored by the nanstd aggregation; the RuntimeWarnings numpy emits for
    all-NaN slices in such resamples are suppressed deliberately.
    """
    n_persons = matrix.shape[0]
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_persons, size=n_persons)
        values = np.asarray(statistic(matrix[idx]), dtype=float)
        samples.append(values)
    stacked = np.vstack(samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(stacked, axis=0, ddof=1)


def _safe_reliability(fn: Callable[[np.ndarray], float], matrix: np.ndarray) -> float:
    """Apply a reliability function, returning NaN where it is undefined."""
    try:
        return float(np.asarray(fn(matrix), dtype=float))
    except ValueError:
        return float("nan")


class CTTAnalysis:
    """Classical test theory analysis of dichotomous response data.

    Parameters
    ----------
    data : ResponseData
        Validated response data.
    n_boot : int, default 1000
        Number of bootstrap resamples for difficulty/discrimination SEs.
    seed : int, optional
        Seed for the bootstrap random generator (reproducibility).

    Examples
    --------
    >>> import numpy as np
    >>> from rasch_per.data import ResponseData
    >>> rd = ResponseData(np.array([[1, 1], [1, 0], [0, 0]]))
    >>> results = CTTAnalysis(rd, n_boot=100, seed=0).run()
    >>> list(results.summary().columns)
    ['item', 'difficulty', 'difficulty_se', 'discrimination', 'discrimination_se']
    """

    def __init__(self, data: ResponseData, n_boot: int = 1000, seed: int | None = None) -> None:
        self._data = data
        self._n_boot = n_boot
        self._seed = seed

    def run(self) -> CTTResults:
        """Compute all CTT statistics.

        Returns
        -------
        CTTResults
        """
        matrix = self._data.to_numpy()
        rng = np.random.default_rng(self._seed)

        difficulty = _item_difficulty(matrix)
        difficulty_se = _bootstrap_se(matrix, _item_difficulty, self._n_boot, rng)
        discrimination = _discrimination(matrix)
        discrimination_se = _bootstrap_se(matrix, _discrimination, self._n_boot, rng)

        reliability = ReliabilityStats(
            cronbach_alpha=_safe_reliability(cronbach_alpha, matrix),
            mcdonald_omega=_safe_reliability(mcdonald_omega, matrix),
            ferguson_delta=_safe_reliability(ferguson_delta, matrix),
        )
        return CTTResults(
            item_names=self._data.item_names,
            difficulty=difficulty,
            difficulty_se=difficulty_se,
            discrimination=discrimination,
            discrimination_se=discrimination_se,
            reliability=reliability,
            n_boot=self._n_boot,
        )
