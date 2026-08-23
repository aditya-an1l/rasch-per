"""Synthetic data generator for testing, demos, and CI.

:func:`simulate_rasch_data` draws person abilities and item difficulties from
configurable distributions, generates responses via the Rasch probability
model plus Bernoulli draws, and can inject known DIF effects and known
misfitting items to validate that the analysis code recovers ground truth.

All example and test data in this package comes from this simulator - no
third-party data is shipped.

Spec reference: section 6.7 of the project build spec.
"""

from __future__ import annotations

__all__ = ["simulate_rasch_data"]


def simulate_rasch_data(
    n_persons: int = 300,
    n_items: int = 20,
    theta_dist: tuple[float, float] | None = None,
    beta_dist: tuple[float, float] | None = None,
    seed: int | None = None,
) -> object:
    """Simulate dichotomous Rasch data (implemented in Phase 2)."""
    raise NotImplementedError("simulate_rasch_data is implemented in Phase 2")
