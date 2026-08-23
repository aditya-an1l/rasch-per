"""Differential Item Functioning (DIF) analysis.

Lord's chi-square test for uniform DIF:

1. Split sample into reference/focal groups via user-supplied labels.
2. Estimate item difficulties separately per group.
3. Link scales with mean/mean linking (Stocking-Lord documented as future work).
4. Per-item Lord's chi-square test of H0: beta_ref = beta_focal.
5. Benjamini-Hochberg correction for multiple comparisons.
6. Effect size on the ETS delta scale: delta = -2.35 * ln(odds ratio),
   binned A (|delta| < 1.0), B (1.0 <= |delta| < 1.5), C (|delta| >= 1.5).

Returns a :class:`DIFResults` dataclass with ``.summary()`` and ``.plot()``.

Spec reference: section 6.4 of the project build spec.
"""

from __future__ import annotations

__all__ = ["DIFAnalysis", "DIFResults"]


class DIFResults:
    """DIF results container (implemented in Phase 4)."""

    def __init__(self) -> None:
        raise NotImplementedError("DIFResults is implemented in Phase 4")


class DIFAnalysis:
    """Lord's chi-square DIF analysis runner (implemented in Phase 4)."""

    def __init__(self, model: object, groups: object, reference: str, focal: str) -> None:
        raise NotImplementedError("DIFAnalysis is implemented in Phase 4")
