"""Test information function, standard errors, and Rasch reliability.

Item information: I_i(theta) = P_i(theta) * (1 - P_i(theta)).
Test information: TIF(theta) = sum_i I_i(theta).
SEM: SEM(theta) = 1 / sqrt(TIF(theta)).
Person separation reliability from true-score variance to observed-score
variance using the SEM function (the Rasch analogue of Cronbach's alpha).

Spec reference: section 6.2 (information) of the project build spec.
"""

from __future__ import annotations

__all__ = ["item_information", "test_information", "sem", "person_separation_reliability"]


def _not_yet(name: str) -> None:
    raise NotImplementedError(f"{name} is implemented in Phase 3")


def item_information(beta: object, theta: object) -> object:
    """Item information I_i(theta) (implemented in Phase 3)."""
    _not_yet("item_information")
    return None


def test_information(betas: object, theta: object) -> object:
    """Test information function (implemented in Phase 3)."""
    _not_yet("test_information")
    return None


def sem(betas: object, theta: object) -> object:
    """Standard error of measurement 1/sqrt(TIF) (implemented in Phase 3)."""
    _not_yet("sem")
    return None


def person_separation_reliability(model: object) -> float:
    """Rasch person separation reliability (implemented in Phase 3)."""
    _not_yet("person_separation_reliability")
    return 0.0
