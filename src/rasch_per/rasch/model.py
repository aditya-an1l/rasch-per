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

__all__ = ["RaschModel", "rasch_probability", "rasch_logit"]


def rasch_probability(theta: object, beta: object) -> object:
    """P(X=1) under the Rasch model (implemented in Phase 2)."""
    raise NotImplementedError("rasch_probability is implemented in Phase 2")


def rasch_logit(theta: object, beta: object) -> object:
    """Logit of the Rasch response probability (implemented in Phase 2)."""
    raise NotImplementedError("rasch_logit is implemented in Phase 2")


class RaschModel:
    """Rasch estimator facade (implemented in Phase 2)."""

    def fit(self, response_data: object, estimator: str = "MML") -> object:
        raise NotImplementedError("RaschModel.fit is implemented in Phase 2")
        return self
