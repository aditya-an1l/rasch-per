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

__all__ = ["cronbach_alpha", "mcdonald_omega", "ferguson_delta"]


def cronbach_alpha(x: object) -> float:
    """Cronbach's alpha (implemented in Phase 3)."""
    raise NotImplementedError("cronbach_alpha is implemented in Phase 3")


def mcdonald_omega(x: object) -> float:
    """McDonald's omega, eigenvalue approximation (implemented in Phase 3)."""
    raise NotImplementedError("mcdonald_omega is implemented in Phase 3")


def ferguson_delta(x: object) -> float:
    """Ferguson's delta (implemented in Phase 3)."""
    raise NotImplementedError("ferguson_delta is implemented in Phase 3")
