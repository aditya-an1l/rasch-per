"""Item and person fit statistics.

Infit and outfit mean-square statistics per item (and optionally per person),
with configurable flagging bounds via presets:

- ``low_stakes``: [0.7, 1.3]
- ``high_stakes``: [0.8, 1.2]
- custom ``(lower, upper)`` tuple

Also Yen's Q3 local-independence statistic:
:func:`compute_q3_matrix` returns an item x item DataFrame of residual
correlations; :func:`flag_q3_violations` flags pairs above a threshold.

The standardized residual for person ``n`` on item ``i`` is

    z_ni = (x_ni - p_ni) / sqrt(p_ni * (1 - p_ni))

with ``p_ni = rasch_probability(theta_n, beta_i)`` and missing responses
excluded. Outfit is the unweighted mean square ``mean(z^2)``; infit is the
information-weighted mean square ``sum(z^2 * p(1-p)) / sum(p(1-p))``.

Spec reference: section 6.2 (fit) of the project build spec.

Note on Q3 convention: this implements the classic Yen (1984) Q3 as the
pairwise Pearson correlation of the standardized residuals above. The project
AGENT.md references a SPEC.md that would pin a specific Q3 "adjustment"
variant; that source document is currently absent from the repo, so the
standard residual-correlation form is used. Flagging this for review.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import expit

if TYPE_CHECKING:
    from rasch_per.rasch.model import RaschModel

__all__ = [
    "FIT_PRESETS",
    "infit_outfit",
    "flag_misfitting_items",
    "compute_q3_matrix",
    "flag_q3_violations",
    "standardized_residuals",
]

FIT_PRESETS: dict[str, tuple[float, float]] = {
    "low_stakes": (0.7, 1.3),
    "high_stakes": (0.8, 1.2),
}


def _resolve_bounds(fit_bounds: str | tuple[float, float]) -> tuple[float, float]:
    """Return (lower, upper) fit bounds from a preset name or tuple."""
    if isinstance(fit_bounds, str):
        if fit_bounds not in FIT_PRESETS:
            raise ValueError(
                f"Unknown fit preset {fit_bounds!r}; use one of "
                f"{sorted(FIT_PRESETS)} or pass an explicit (lower, upper) tuple."
            )
        return FIT_PRESETS[fit_bounds]
    lower, upper = fit_bounds
    if not (lower < 1.0 < upper):
        raise ValueError("fit bounds must straddle 1.0 (e.g. (0.7, 1.3))")
    return float(lower), float(upper)


def _model_arrays(model: RaschModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract (matrix, betas, thetas, item_names) from a fitted RaschModel."""
    try:
        betas = np.asarray(model.item_difficulties.to_numpy(), dtype=float)
        thetas = np.asarray(model.person_abilities.to_numpy(), dtype=float)
        matrix = np.asarray(model.responses, dtype=float)
        item_names = list(model.item_names)
    except AttributeError as exc:  # pragma: no cover - defensive typing guard
        raise TypeError(
            "fit functions expect a fitted RaschModel (or duck-typed object with "
            "item_difficulties, person_abilities, responses, item_names)"
        ) from exc
    return matrix, betas, thetas, item_names


def standardized_residuals(model: RaschModel) -> np.ndarray:
    """Standardized Rasch residuals (persons x items), NaN where missing.

    Parameters
    ----------
    model : fitted RaschModel
        A model that has been `.fit()`; exposes ``item_difficulties``,
        ``person_abilities`` and ``responses``.

    Returns
    -------
    numpy.ndarray
        Persons x items array of z-scores ``(x - p) / sqrt(p(1-p))``; entries
        for unobserved (NaN) responses are NaN.
    """
    matrix, betas, thetas, _ = _model_arrays(model)
    p = expit(thetas[:, None] - betas[None, :])
    variance = p * (1.0 - p)
    z = (matrix - p) / np.sqrt(variance)
    z[~np.isfinite(z)] = np.nan  # NaN responses and any numerical edge cases
    return z


def infit_outfit(model: RaschModel) -> pd.DataFrame:
    """Infit and outfit mean-square fit statistics per item.

    Outfit is the unweighted mean of squared standardized residuals; infit
    weights each squared residual by the model-predicted item variance
    ``p(1-p)`` at that person. Values near 1.0 indicate productive model fit.

    Parameters
    ----------
    model : fitted RaschModel
        Model with ``item_difficulties``, ``person_abilities`` and
        ``responses`` available.

    Returns
    -------
    pandas.DataFrame
        Indexed by item name with columns ``infit`` and ``outfit``.

    Examples
    --------
    >>> from rasch_per.data import ResponseData
    >>> from rasch_per.rasch import RaschModel
    >>> df = ResponseData(pd.DataFrame({"a": [1, 0], "b": [0, 1]}))
    >>> m = RaschModel().fit(df, estimator="JML")
    >>> fit = infit_outfit(m)
    >>> sorted(fit.columns)
    ['infit', 'outfit']
    """
    matrix, betas, thetas, item_names = _model_arrays(model)
    p = expit(thetas[:, None] - betas[None, :])
    observed = np.isfinite(matrix)
    z = np.where(observed, (matrix - p) / np.sqrt(p * (1.0 - p)), np.nan)

    z2 = np.where(observed, z**2, np.nan)
    info = np.where(observed, p * (1.0 - p), np.nan)

    outfit = np.nanmean(z2, axis=0)
    infit = np.nansum(z2 * info, axis=0) / np.nansum(info, axis=0)

    return pd.DataFrame({"infit": infit, "outfit": outfit}, index=item_names)


def flag_misfitting_items(
    model: RaschModel, fit_bounds: str | tuple[float, float] = "low_stakes"
) -> pd.DataFrame:
    """Flag items whose infit/outfit mean squares fall outside fit bounds.

    Parameters
    ----------
    model : fitted RaschModel
        Model with item and person estimates.
    fit_bounds : str or (float, float), default "low_stakes"
        Preset name ("low_stakes" = [0.7, 1.3], "high_stakes" = [0.8, 1.2])
        or an explicit ``(lower, upper)`` tuple. Bounds must straddle 1.0.

    Returns
    -------
    pandas.DataFrame
        Indexed by item name with columns ``infit``, ``outfit``,
        ``infit_misfit`` and ``outfit_misfit`` (booleans; True when the
        statistic is outside the bounds).
    """
    lower, upper = _resolve_bounds(fit_bounds)
    fit = infit_outfit(model)
    result = fit.copy()
    result["infit_misfit"] = (fit["infit"] < lower) | (fit["infit"] > upper)
    result["outfit_misfit"] = (fit["outfit"] < lower) | (fit["outfit"] > upper)
    return result


def compute_q3_matrix(model: RaschModel) -> pd.DataFrame:
    """Yen's Q3 residual correlation matrix between items.

    Computes the pairwise Pearson correlation of the standardized residuals
    across persons (pairwise deletion of missing responses). Large positive
    off-diagonal values indicate local dependence beyond the Rasch dimension.

    Parameters
    ----------
    model : fitted RaschModel
        Model with item and person estimates and responses.

    Returns
    -------
    pandas.DataFrame
        Item x item symmetric correlation matrix indexed by item name.

    Examples
    --------
    >>> from rasch_per.data import ResponseData
    >>> from rasch_per.rasch import RaschModel
    >>> df = ResponseData(pd.DataFrame({"a": [1, 0], "b": [0, 1]}))
    >>> m = RaschModel().fit(df, estimator="JML")
    >>> q3 = compute_q3_matrix(m)
    >>> q3.shape
    (2, 2)
    """
    z = standardized_residuals(model)
    frame = pd.DataFrame(z, columns=_model_arrays(model)[3])
    return frame.corr()


def flag_q3_violations(q3: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """Flag Q3 item pairs whose absolute residual correlation exceeds a threshold.

    Parameters
    ----------
    q3 : pandas.DataFrame
        Item x item Q3 matrix from :func:`compute_q3_matrix`.
    threshold : float, default 0.2
        Absolute correlation above which a pair is flagged.

    Returns
    -------
    pandas.DataFrame
        Columns ``item_a``, ``item_b``, ``q3`` for flagged pairs (upper
        triangle only, ``item_a`` < ``item_b``), sorted by absolute
        correlation descending.
    """
    items = list(q3.index)
    rows: list[dict[str, object]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            value = float(q3.to_numpy()[i, j])
            if abs(value) > threshold:
                rows.append({"item_a": items[i], "item_b": items[j], "q3": value})
    flagged = pd.DataFrame(rows, columns=["item_a", "item_b", "q3"])
    if not flagged.empty:
        flagged = flagged.sort_values("q3", key=lambda s: s.abs(), ascending=False).reset_index(
            drop=True
        )
    return flagged
