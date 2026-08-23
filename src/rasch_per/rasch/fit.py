"""Item and person fit statistics.

Infit and outfit mean-square statistics per item (and optionally per person),
with configurable flagging bounds via presets:

- ``low_stakes``: [0.7, 1.3]
- ``high_stakes``: [0.8, 1.2]
- custom ``(lower, upper)`` tuple

Also Yen's Q3 local-independence statistic:
``compute_q3_matrix(model)`` returns an item x item DataFrame of residual
correlations; ``flag_q3_violations(threshold=0.2)`` flags violations.

Spec reference: section 6.2 (fit) of the project build spec.
"""

from __future__ import annotations

__all__ = ["infit_outfit", "flag_misfitting_items", "compute_q3_matrix", "flag_q3_violations"]


def _not_yet(name: str) -> None:
    raise NotImplementedError(f"{name} is implemented in Phase 3")


def infit_outfit(model: object) -> object:
    """Infit/outfit mean-square table (implemented in Phase 3)."""
    _not_yet("infit_outfit")
    return None


def flag_misfitting_items(
    model: object, fit_bounds: str | tuple[float, float] = "low_stakes"
) -> object:
    """Flag items outside fit bounds (implemented in Phase 3)."""
    _not_yet("flag_misfitting_items")
    return None


def compute_q3_matrix(model: object) -> object:
    """Yen's Q3 residual correlation matrix (implemented in Phase 3)."""
    _not_yet("compute_q3_matrix")
    return None


def flag_q3_violations(q3: object, threshold: float = 0.2) -> object:
    """Flag Q3 pairs above threshold (implemented in Phase 3)."""
    _not_yet("flag_q3_violations")
    return None
