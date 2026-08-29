"""Rasch model core.

Dichotomous Rasch model:

    P(X_ni = 1) = exp(theta_n - beta_i) / (1 + exp(theta_n - beta_i))
    logit(X_ni) = theta_n - beta_i

Provides vectorized ``rasch_probability(theta, beta)`` and
``rasch_logit(theta, beta)``, broadcasting over arrays of theta and beta, plus
the :class:`RaschModel` estimator facade with ``.fit(response_data,
estimator="MML"|"JML")`` exposing ``item_difficulties``, ``person_abilities``
and ``standard_errors``.

Spec reference: section 6.2 of the project build spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from rasch_per.data import ResponseData
from rasch_per.rasch.estimation import EstimationResult, fit_jml, fit_mml

__all__ = ["RaschModel", "rasch_probability", "rasch_logit"]


def rasch_logit(theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Logit of the Rasch response probability: theta - beta.

    Parameters
    ----------
    theta : array-like
        Person ability parameter(s) on the logit scale.
    beta : array-like
        Item difficulty parameter(s) on the logit scale.

    Returns
    -------
    numpy.ndarray
        Broadcast difference ``theta - beta``.

    Examples
    --------
    >>> rasch_logit(1.5, 0.5)
    1.0
    """
    return np.asarray(theta, dtype=float) - np.asarray(beta, dtype=float)


def rasch_probability(theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Probability of a correct response under the Rasch model.

    P(X=1) = exp(theta - beta) / (1 + exp(theta - beta)), computed via the
    numerically stable logistic sigmoid.

    Parameters
    ----------
    theta : array-like
        Person ability parameter(s).
    beta : array-like
        Item difficulty parameter(s).

    Returns
    -------
    numpy.ndarray
        Broadcast probabilities in (0, 1). Shape is the broadcast of
        ``theta`` and ``beta``.

    Examples
    --------
    >>> float(rasch_probability(0.0, 0.0))
    0.5
    >>> float(rasch_probability(2.0, 2.0))
    0.5
    """
    return expit(rasch_logit(theta, beta))


class RaschModel:
    """Fitted Rasch model facade.

    Typical usage::

        model = RaschModel().fit(response_data, estimator="MML")
        model.item_difficulties   # pd.Series indexed by item name
        model.person_abilities    # pd.Series indexed by person_id
        model.standard_errors     # dict of SE Series

    Attributes are only available after calling :meth:`fit`.
    """

    def __init__(self) -> None:
        self._result: EstimationResult | None = None
        self._item_names: list[str] = []
        self._person_ids: list = []
        self._matrix: np.ndarray | None = None

    def fit(
        self,
        response_data: ResponseData,
        estimator: str = "MML",
        max_iter: int = 500,
        tol: float = 1e-6,
        n_nodes: int = 61,
        sigma: float = 1.0,
    ) -> RaschModel:
        """Estimate item difficulties and person abilities.

        Parameters
        ----------
        response_data : ResponseData
            Validated dichotomous response data.
        estimator : {"MML", "JML"}, default "MML"
            MML (marginal maximum likelihood with theta ~ N(0, sigma^2)
            integrated by Gauss-Hermite quadrature) is the default, mirroring
            R's TAM. JML (joint maximum likelihood) is also available.
        max_iter : int
            Maximum EM / alternating iterations.
        tol : float
            Convergence tolerance on parameter change.
        n_nodes : int
            Number of Gauss-Hermite quadrature nodes (MML only).
        sigma : float
            Fixed population ability standard deviation (MML only);
            fixing it identifies the scale together with mean(beta) = 0.

        Returns
        -------
        RaschModel
            The fitted model (self), for chaining.
        """
        matrix = response_data.to_numpy()
        if estimator == "MML":
            result = fit_mml(matrix, max_iter=max_iter, tol=tol, n_nodes=n_nodes, sigma=sigma)
        elif estimator == "JML":
            result = fit_jml(matrix, max_iter=max_iter, tol=tol)
        else:
            raise ValueError(f"Unknown estimator {estimator!r}; use 'MML' or 'JML'")
        self._result = result
        self._matrix = matrix
        self._item_names = response_data.item_names
        self._person_ids = response_data.person_ids
        return self

    def _require_fitted(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("Call .fit() before accessing estimates")
        return self._result

    @property
    def item_difficulties(self) -> pd.Series:
        """Estimated item difficulties indexed by item name."""
        result = self._require_fitted()
        return pd.Series(result.betas, index=self._item_names, name="beta")

    @property
    def person_abilities(self) -> pd.Series:
        """Estimated person abilities indexed by person_id."""
        result = self._require_fitted()
        return pd.Series(result.thetas, index=self._person_ids, name="theta")

    @property
    def standard_errors(self) -> dict[str, pd.Series]:
        """Standard errors for both parameter sets."""
        result = self._require_fitted()
        return {
            "item_difficulty": pd.Series(result.se_beta, index=self._item_names, name="se_beta"),
            "person_ability": pd.Series(result.se_theta, index=self._person_ids, name="se_theta"),
        }

    @property
    def responses(self) -> np.ndarray:
        """Response matrix used in the most recent :meth:`fit` (float, NaN missing)."""
        if self._matrix is None:
            raise RuntimeError("Call .fit() before accessing responses")
        return self._matrix

    @property
    def item_names(self) -> list[str]:
        """Item names from the data used in the most recent :meth:`fit`."""
        if not self._item_names:
            raise RuntimeError("Call .fit() before accessing item_names")
        return list(self._item_names)

    def fit_statistics(self) -> pd.DataFrame:
        """Per-item infit and outfit mean-square fit statistics.

        Returns a DataFrame indexed by item name with ``infit`` and ``outfit``
        columns (see :func:`rasch_per.rasch.fit.infit_outfit`).

        Returns
        -------
        pandas.DataFrame
        """
        from rasch_per.rasch.fit import infit_outfit

        return infit_outfit(self)
