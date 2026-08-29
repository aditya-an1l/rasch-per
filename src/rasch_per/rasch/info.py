"""Test information function, standard errors, and Rasch reliability.

Item information: I_i(theta) = P_i(theta) * (1 - P_i(theta)).
Test information: TIF(theta) = sum_i I_i(theta).
SEM: SEM(theta) = 1 / sqrt(TIF(theta)).
Person separation reliability from true-score variance to observed-score
variance using the SEM function (the Rasch analogue of Cronbach's alpha).

Spec reference: section 6.2 (information) of the project build spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import expit

if TYPE_CHECKING:
    from rasch_per.rasch.model import RaschModel

__all__ = ["item_information", "test_information", "sem", "person_separation_reliability"]


def _scalar_or_array(result: np.ndarray, is_scalar: bool) -> np.ndarray | float:
    """Return a Python float when ``theta`` was a scalar, else the array."""
    if is_scalar:
        return float(result[0])
    return result


def item_information(beta: float, theta: object) -> np.ndarray | float:
    """Item information function I_i(theta) = P_i(1 - P_i).

    Parameters
    ----------
    beta : float
        Difficulty of the single item whose information is computed.
    theta : array-like or scalar
        Ability point(s) at which to evaluate the information.

    Returns
    -------
    float or numpy.ndarray
        Information at each ``theta``; a Python float when ``theta`` is a
        scalar, otherwise a 1-D array.

    Examples
    --------
    >>> float(item_information(0.0, 0.0))
    0.25
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
    p = expit(theta_arr - float(beta))
    return _scalar_or_array(p * (1.0 - p), np.asarray(theta).ndim == 0)


def test_information(betas: object, theta: object) -> np.ndarray | float:
    """Test information function (sum of item information) over items.

    Parameters
    ----------
    betas : array-like
        Difficulties of all items.
    theta : array-like or scalar
        Ability point(s) at which to evaluate the TIF.

    Returns
    -------
    float or numpy.ndarray
        Total information at each ``theta``; a Python float when ``theta`` is
        a scalar, otherwise a 1-D array.
    """
    betas_arr = np.atleast_1d(np.asarray(betas, dtype=float))
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
    d = theta_arr[:, None] - betas_arr[None, :]
    p = expit(d)
    return _scalar_or_array(np.sum(p * (1.0 - p), axis=1), np.asarray(theta).ndim == 0)


def sem(betas: object, theta: object) -> np.ndarray | float:
    """Standard error of measurement: SEM(theta) = 1 / sqrt(TIF(theta).

    Parameters
    ----------
    betas : array-like
        Difficulties of all items.
    theta : array-like or scalar
        Ability point(s) at which to evaluate the SEM.

    Returns
    -------
    float or numpy.ndarray
        Standard error at each ``theta``; a Python float when ``theta`` is a
        scalar, otherwise a 1-D array.
    """
    tif = test_information(betas, theta)
    return 1.0 / np.sqrt(tif)


def person_separation_reliability(model: RaschModel) -> float:
    """Rasch person separation reliability.

    Estimates the proportion of observed person-ability variance that is true
    (signal) rather than measurement error (noise):

        rel = (var(theta) - mean(se_theta^2)) / var(theta)

    where ``theta`` are the person ability estimates and ``se_theta`` are their
    standard errors. This is the Rasch analogue of Cronbach's alpha for person
    separation.

    Parameters
    ----------
    model : fitted RaschModel
        Model exposing ``person_abilities`` and ``standard_errors``.

    Returns
    -------
    float
        Reliability in [0, 1] (clamped; can be slightly negative due to
        sampling noise when separation is very poor).

    Examples
    --------
    >>> from rasch_per.data import ResponseData
    >>> from rasch_per.rasch import RaschModel
    >>> df = ResponseData([[1, 0, 1], [0, 1, 1]])
    >>> m = RaschModel().fit(df, estimator="JML")
    >>> 0.0 <= person_separation_reliability(m) <= 1.0
    True
    """
    try:
        thetas = np.asarray(model.person_abilities.to_numpy(), dtype=float)
        se = np.asarray(model.standard_errors["person_ability"].to_numpy(), dtype=float)
    except (AttributeError, KeyError, TypeError) as exc:  # pragma: no cover
        raise TypeError(
            "person_separation_reliability expects a fitted RaschModel with "
            "person_abilities and standard_errors"
        ) from exc

    valid = np.isfinite(thetas) & np.isfinite(se)
    if valid.sum() < 3:
        return float("nan")
    theta_v = thetas[valid]
    se_v = se[valid]

    var_theta = float(np.var(theta_v, ddof=1))
    if var_theta <= 0.0:
        return 0.0
    mean_se2 = float(np.mean(se_v**2))
    reliability = (var_theta - mean_se2) / var_theta
    return float(min(1.0, max(0.0, reliability)))
