"""Differential Item Functioning (DIF) analysis.

Lord's chi-square test for uniform DIF:

1. Split the sample into reference and focal groups via user-supplied labels.
2. Estimate item difficulties separately per group (Rasch calibration).
3. Place the two sets of difficulties on a common scale with mean/mean
   linking (Stocking-Lord is documented as future work).
4. Per-item Lord's chi-square (1 df, Wald form) tests H0: beta_ref = beta_focal.
5. Benjamini-Hochberg correction controls the false discovery rate across items.
6. Effect size on the ETS delta scale: delta = -2.35 * ln(odds ratio),
   binned A (|delta| < 1.0), B (1.0 <= |delta| < 1.5), C (|delta| >= 1.5).

Returns a :class:`DIFResults` container with ``.summary()`` (the per-item
table) and ``.plot()`` (deferred to the plotting module, Phase 5).

Spec reference: section 6.4 of the project build spec.

Note on conventions: the ETS delta sign and the A/B/C bin thresholds follow
the values pinned in this module's stub (treated as the spec fragment). The
project AGENT.md references a SPEC.md for the ETS delta sign; that source file
is currently absent from the repo, so this implementation follows the stub
exactly and the gap is flagged in GRILL.md for later re-verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from rasch_per.rasch.estimation import fit_jml, fit_mml

if TYPE_CHECKING:
    from rasch_per.rasch.model import RaschModel

__all__ = ["DIFAnalysis", "DIFResults", "benjamini_hochberg", "ets_delta_class"]

_ETS_A = 1.0
_ETS_B = 1.5


def benjamini_hochberg(p_values: object, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values) for FDR control.

    Parameters
    ----------
    p_values : array-like
        Raw p-values, one per test.
    alpha : float, default 0.05
        Target false-discovery rate.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values (q-values) in the same order as the input, each in
        [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    m = p.shape[0]
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    # BH adjusted p: min over tail of (m / rank) * p_ranked, then monotone.
    scaled = ranked * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.clip(scaled, 0.0, 1.0)
    return out


def ets_delta_class(ets_delta: float) -> str:
    """Bin an ETS delta into A (negligible) / B (moderate) / C (large).

    Uses |delta| thresholds 1.0 and 1.5 per the project spec fragment. A NaN
    input (undefined proportion) returns ``"NA"``.
    """
    if not np.isfinite(ets_delta):
        return "NA"
    magnitude = abs(ets_delta)
    if magnitude < _ETS_A:
        return "A"
    if magnitude < _ETS_B:
        return "B"
    return "C"


@dataclass
class DIFResults:
    """Container for DIF analysis output.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per item with columns: ``beta_ref``, ``beta_focal``,
        ``dif`` (linked difficulty difference), ``lord_chi2``, ``p_value``,
        ``bh_q`` (adjusted p), ``flag`` (significant after BH), ``ets_delta``,
        ``ets_class``.
    alpha : float
        The false-discovery rate used for flagging.
    """

    table: pd.DataFrame
    alpha: float

    def summary(self) -> pd.DataFrame:
        """Return the per-item DIF summary table."""
        return self.table

    def plot(self) -> object:
        """DIF plot (deferred to the plotting module, Phase 5)."""
        raise NotImplementedError("DIF plots are implemented in Phase 5 (plots.py)")


class DIFAnalysis:
    """Lord's chi-square DIF analysis runner.

    Parameters
    ----------
    model : fitted RaschModel
        A model fitted on the full sample; supplies ``responses``,
        ``item_names`` and ``person_ids``.
    groups : array-like
        Group label per person, aligned to ``model.person_ids`` order.
    reference : label
        Group label treated as the reference group.
    focal : label
        Group label treated as the focal group.
    estimator : {"MML", "JML"}, default "MML"
        Estimator used for the per-group Rasch calibrations.
    alpha : float, default 0.05
        False-discovery rate for Benjamini-Hochberg flagging.
    """

    def __init__(
        self,
        model: RaschModel,
        groups: object,
        reference: object,
        focal: object,
        estimator: str = "MML",
        alpha: float = 0.05,
    ) -> None:
        self._model: RaschModel = model
        self._groups = np.asarray(groups)
        self._reference = reference
        self._focal = focal
        self._estimator = estimator
        self._alpha = alpha

    def _group_calibration(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit Rasch on the subgroup rows; return (betas, se_betas)."""
        matrix = np.asarray(self._model.responses, dtype=float)[mask]
        if self._estimator == "MML":
            result = fit_mml(matrix)
        elif self._estimator == "JML":
            result = fit_jml(matrix)
        else:
            raise ValueError(f"Unknown estimator {self._estimator!r}; use 'MML' or 'JML'")
        return result.betas, result.se_beta

    def analyze(self) -> DIFResults:
        """Run the full DIF pipeline and return a :class:`DIFResults`."""
        item_names = list(self._model.item_names)
        matrix = np.asarray(self._model.responses, dtype=float)
        ref_mask = self._groups == self._reference
        focal_mask = self._groups == self._focal
        if not ref_mask.any() or not focal_mask.any():
            raise ValueError("Both reference and focal groups must be present")

        betas_ref, se_ref = self._group_calibration(ref_mask)
        betas_focal, se_focal = self._group_calibration(focal_mask)

        # Mean/mean linking: align focal scale to the reference scale.
        linking = float(np.mean(betas_ref) - np.mean(betas_focal))
        betas_focal_linked = betas_focal + linking
        dif = betas_focal_linked - betas_ref

        # Lord's chi-square (1 df, Wald form) on the linked difficulty gap.
        denom = se_ref**2 + se_focal**2
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.where(denom > 0, dif**2 / np.where(denom > 0, denom, np.nan), np.nan)
        p_value = np.where(np.isfinite(chi2), stats.chi2.sf(chi2, df=1), np.nan)

        # ETS delta from observed group proportions correct.
        p_ref = np.nanmean(np.where(ref_mask[:, None], matrix, np.nan), axis=0)
        p_focal = np.nanmean(np.where(focal_mask[:, None], matrix, np.nan), axis=0)

        def _ets(p_f: float, p_r: float) -> float:
            if not (0.0 < p_f < 1.0 and 0.0 < p_r < 1.0):
                return float("nan")
            odds_f = p_f / (1.0 - p_f)
            odds_r = p_r / (1.0 - p_r)
            if odds_r <= 0:
                return float("nan")
            return -2.35 * np.log(odds_f / odds_r)

        ets_delta = np.array(
            [_ets(pf, pr) for pf, pr in zip(p_focal, p_ref, strict=False)], dtype=float
        )
        ets_class = np.array([ets_delta_class(v) for v in ets_delta], dtype=object)

        bh_q = benjamini_hochberg(p_value, alpha=self._alpha)
        flag = np.isfinite(bh_q) & (bh_q < self._alpha)

        table = pd.DataFrame(
            {
                "item": item_names,
                "beta_ref": betas_ref,
                "beta_focal": betas_focal_linked,
                "dif": dif,
                "lord_chi2": chi2,
                "p_value": p_value,
                "bh_q": bh_q,
                "flag": flag,
                "ets_delta": ets_delta,
                "ets_class": ets_class,
            }
        ).set_index("item")

        return DIFResults(table=table, alpha=self._alpha)
