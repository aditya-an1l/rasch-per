"""Parameter estimation for the Rasch model.

Two estimators:

- **JML** (Joint Maximum Likelihood): alternating conditional MLE for theta
  and beta until convergence; identifiability constraint mean(beta) = 0;
  extreme scores handled with the standard +/-0.3 logit adjustment.
- **MML** (Marginal Maximum Likelihood, default): theta ~ N(0, sigma^2),
  integrated out via Gauss-Hermite quadrature; item parameters estimated by
  EM. Mirrors R's TAM default.

Standard errors come from observed/expected information matrices.

Spec reference: section 6.2 (estimation) of the project build spec.
"""

from __future__ import annotations

__all__ = ["fit_jml", "fit_mml"]


def fit_jml(matrix: object, max_iter: int = 1000, tol: float = 1e-6) -> object:
    """Joint maximum likelihood estimation (implemented in Phase 2)."""
    raise NotImplementedError("fit_jml is implemented in Phase 2")


def fit_mml(matrix: object, max_iter: int = 1000, tol: float = 1e-6) -> object:
    """Marginal maximum likelihood estimation via Gauss-Hermite EM (Phase 2)."""
    raise NotImplementedError("fit_mml is implemented in Phase 2")
