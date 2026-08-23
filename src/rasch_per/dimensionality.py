"""Dimensionality diagnostics: Principal Components Analysis of Residuals.

After fitting the Rasch model, standardized residuals are extracted and PCA
is run on them; the eigenvalue of the first contrast is reported. An
eigenvalue above 2.0 suggests a possible second dimension (advisory flag,
never an automatic failure).

Optional single-factor CFA via semopy is an extra dependency
(``pip install rasch-per[cfa]``), not part of core v1.

Spec reference: section 6.3 of the project build spec.
"""

from __future__ import annotations

__all__ = ["run_pcar"]


def run_pcar(model: object) -> object:
    """PCAR first-contrast eigenvalue (implemented in Phase 3)."""
    raise NotImplementedError("run_pcar is implemented in Phase 3")
