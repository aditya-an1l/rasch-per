"""Matplotlib plotting functions for Rasch and CTT outputs.

Every function returns a :class:`matplotlib.figure.Figure` and never calls
``plt.show()`` internally. This module sets the Agg backend only when running
headless (no ``DISPLAY`` and backend not already Agg) so an existing user
backend is never clobbered.

Planned plots: ICC with empirical overlay, Wright map, test information/SEM,
CTT difficulty/discrimination bars, DIF contrasts, group ability distributions.

Spec reference: section 6.5 of the project build spec.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() not in ("agg",):
    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rasch_per.rasch.info import sem, test_information  # noqa: E402
from rasch_per.rasch.model import rasch_probability  # noqa: E402

if TYPE_CHECKING:
    from rasch_per.ctt import CTTResults
    from rasch_per.dif import DIFResults
    from rasch_per.rasch.model import RaschModel

__all__ = [
    "plot_icc",
    "plot_wright_map",
    "plot_test_information",
    "plot_item_difficulty_bar",
    "plot_item_discrimination_bar",
    "plot_dif_contrasts",
    "plot_group_ability_distributions",
]

_THETA_GRID = np.linspace(-3.5, 3.5, 200)


def plot_icc(model: RaschModel, item: str) -> object:
    """Item characteristic curve with an empirical overlay for ``item``."""
    beta = float(model.item_difficulties[item])
    theta = np.asarray(model.person_abilities.to_numpy(), dtype=np.float64)
    item_idx = model.item_names.index(item)
    responses = np.asarray(model.responses, dtype=np.float64)[:, item_idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(_THETA_GRID, rasch_probability(_THETA_GRID, np.asarray(beta)), label="Model ICC")
    valid = np.isfinite(theta) & np.isfinite(responses)
    if valid.any():
        bins = np.linspace(-3.5, 3.5, 11)
        centers = 0.5 * (bins[:-1] + bins[1:])
        emp = np.full(len(centers), np.nan)
        for b in range(len(bins) - 1):
            mask = valid & (theta >= bins[b]) & (theta < bins[b + 1])
            if mask.sum() > 0:
                emp[b] = float(np.nanmean(responses[mask]))
        ax.plot(centers, emp, "o", label="Empirical")
    ax.set_xlabel("Ability (theta)")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"ICC - {item}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_wright_map(model: RaschModel) -> object:
    """Wright map: person-ability histogram with item-difficulty markers."""
    theta = np.asarray(model.person_abilities.to_numpy(), dtype=np.float64)
    betas = np.asarray(model.item_difficulties.to_numpy(), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.hist(theta, bins=30, orientation="horizontal", color="steelblue", alpha=0.7)
    ax.set_xlabel("Person count")
    ax.set_ylabel("Ability / difficulty (logits)")
    ax2 = ax.twiny()
    for b in betas:
        ax2.plot([1.05, 1.25], [b, b], color="darkred")
    ax2.set_xlim(0, 1.3)
    ax2.set_xticks([])
    ax.set_title("Wright map")
    fig.tight_layout()
    return fig


def plot_test_information(model: RaschModel) -> object:
    """Test information curve with the corresponding SEM curve."""
    betas = np.asarray(model.item_difficulties.to_numpy(), dtype=np.float64)
    info = test_information(betas, _THETA_GRID)
    se = sem(betas, _THETA_GRID)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(_THETA_GRID, info, label="Test information")
    ax.plot(_THETA_GRID, se, label="SEM", linestyle="--")
    ax.set_xlabel("Ability (theta)")
    ax.set_ylabel("Information / SEM")
    ax.set_title("Test information and SEM")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_item_difficulty_bar(ctt_results: CTTResults) -> object:
    """CTT difficulty bar chart with standard-error error bars."""
    table = ctt_results.summary()
    items = list(table.index)
    difficulty = table["difficulty"].to_numpy(dtype=float)
    se = table.get("difficulty_se")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(items)), difficulty, color="steelblue")
    if se is not None:
        ax.errorbar(
            range(len(items)),
            difficulty,
            yerr=se.to_numpy(dtype=float),
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(items, rotation=45, ha="right")
    ax.set_ylabel("Difficulty (proportion correct)")
    ax.set_title("Item difficulty")
    fig.tight_layout()
    return fig


def plot_item_discrimination_bar(ctt_results: CTTResults) -> object:
    """CTT discrimination bar chart with a 0.2 reference line."""
    table = ctt_results.summary()
    items = list(table.index)
    discrimination = table["discrimination"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(items)), discrimination, color="seagreen")
    ax.axhline(0.2, color="red", linestyle="--", label="0.2 reference")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(items, rotation=45, ha="right")
    ax.set_ylabel("Discrimination (point-biserial r)")
    ax.set_title("Item discrimination")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_dif_contrasts(dif_results: DIFResults) -> object:
    """DIF contrast scatter with ETS A/B/C threshold lines."""
    table = dif_results.summary()
    x = table["dif"].to_numpy(dtype=float)
    flagged = table["flag"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x[~flagged], np.arange(len(x))[~flagged], c="gray", label="Not flagged")
    ax.scatter(x[flagged], np.arange(len(x))[flagged], c="red", label="Flagged")
    ax.axvline(1.0, color="orange", linestyle="--")
    ax.axvline(-1.0, color="orange", linestyle="--")
    ax.axvline(1.5, color="darkred", linestyle="--")
    ax.axvline(-1.5, color="darkred", linestyle="--")
    ax.set_xlabel("DIF contrast (logits, linked)")
    ax.set_ylabel("Item index")
    ax.set_title("DIF contrasts")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_group_ability_distributions(model: RaschModel, groups: object) -> object:
    """Overlaid ability histograms per group."""
    theta = np.asarray(model.person_abilities, dtype=float)
    labels = np.asarray(groups)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in np.unique(labels):
        mask = labels == label
        ax.hist(theta[mask], bins=25, alpha=0.5, label=str(label))
    ax.set_xlabel("Ability (theta)")
    ax.set_ylabel("Person count")
    ax.set_title("Group ability distributions")
    ax.legend()
    fig.tight_layout()
    return fig
