"""Matplotlib plotting functions for Rasch and CTT outputs.

Every function returns a :class:`matplotlib.figure.Figure` and never calls
``plt.show()`` internally. The Agg backend is used for headless report
generation, guarded so an existing user backend is not clobbered.

Planned plots: ICC with empirical overlay, Wright map, test information/SEM,
CTT difficulty/discrimination bars, DIF contrasts, group ability distributions.

Spec reference: section 6.5 of the project build spec.
"""

from __future__ import annotations

__all__ = [
    "plot_icc",
    "plot_wright_map",
    "plot_test_information",
    "plot_item_difficulty_bar",
    "plot_item_discrimination_bar",
    "plot_dif_contrasts",
    "plot_group_ability_distributions",
]


def _not_yet(name: str) -> None:
    raise NotImplementedError(f"{name} is implemented in Phase 5")


def plot_icc(model: object, item: str) -> object:
    """Item characteristic curve with empirical overlay (implemented in Phase 5)."""
    _not_yet("plot_icc")
    return None


def plot_wright_map(model: object) -> object:
    """Wright map (implemented in Phase 5)."""
    _not_yet("plot_wright_map")
    return None


def plot_test_information(model: object) -> object:
    """Test information + SEM plot (implemented in Phase 5)."""
    _not_yet("plot_test_information")
    return None


def plot_item_difficulty_bar(ctt_results: object) -> object:
    """CTT difficulty bar chart with 95% CI (implemented in Phase 5)."""
    _not_yet("plot_item_difficulty_bar")
    return None


def plot_item_discrimination_bar(ctt_results: object) -> object:
    """CTT discrimination bar chart with 0.2 reference line (Phase 5)."""
    _not_yet("plot_item_discrimination_bar")
    return None


def plot_dif_contrasts(dif_results: object) -> object:
    """DIF contrast scatter with ETS thresholds (implemented in Phase 5)."""
    _not_yet("plot_dif_contrasts")
    return None


def plot_group_ability_distributions(model: object, groups: object) -> object:
    """Overlaid theta histograms per group (implemented in Phase 5)."""
    _not_yet("plot_group_ability_distributions")
    return None
