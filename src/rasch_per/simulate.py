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

import numpy as np
import pandas as pd
from scipy.special import expit

__all__ = ["simulate_rasch_data"]


def simulate_rasch_data(
    n_persons: int = 300,
    n_items: int = 20,
    theta_dist: tuple[float, float] | None = None,
    beta_dist: tuple[float, float] | None = None,
    seed: int | None = None,
    groups: np.ndarray | list | None = None,
    focal_label: str | float = "focal",
    dif_effects: dict[int, float] | None = None,
    misfit_items: list[int] | None = None,
    misfit_prob: float = 0.2,
    return_parameters: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate synthetic dichotomous responses under the Rasch model.

    Parameters
    ----------
    n_persons : int
        Number of simulated respondents.
    n_items : int
        Number of simulated items.
    theta_dist : (mean, sd), optional
        Normal distribution of person abilities. Default N(0, 1).
    beta_dist : (mean, sd), optional
        Normal distribution of item difficulties. Default N(0, 1.5).
    seed : int, optional
        Random seed for reproducibility.
    groups : array-like, optional
        Group label per person (length ``n_persons``), e.g. for DIF testing.
    focal_label : object, default "focal"
        Which group label is the DIF focal group.
    dif_effects : dict[int, float], optional
        Item index -> logit shift applied to the focal group on that item.
        Positive values make the item harder for the focal group.
    misfit_items : list of int, optional
        Item indices where model fit is deliberately violated by randomly
        flipping a fraction of drawn responses (inflates outfit).
    misfit_prob : float, default 0.2
        Flip probability for misfitting items.
    return_parameters : bool, default False
        If True, return ``(dataframe, parameters)`` where parameters holds
        the ground-truth ``theta``, ``beta`` arrays.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, dict)
        A frame with a ``person_id`` column plus one 0/1 column per item;
        optionally paired with the ground-truth parameters.

    Examples
    --------
    >>> df, params = simulate_rasch_data(n_persons=50, n_items=5, seed=0,
    ...                                  return_parameters=True)
    >>> df.shape
    (50, 6)
    >>> sorted(params.keys())
    ['beta', 'theta']
    """
    if n_persons < 1 or n_items < 1:
        raise ValueError("n_persons and n_items must be positive")
    rng = np.random.default_rng(seed)

    theta_mean, theta_sd = theta_dist if theta_dist is not None else (0.0, 1.0)
    beta_mean, beta_sd = beta_dist if beta_dist is not None else (0.0, 1.5)
    theta = rng.normal(theta_mean, theta_sd, size=n_persons)
    beta = rng.normal(beta_mean, beta_sd, size=n_items)

    # Inject known uniform DIF: shift effective difficulty for focal members.
    if dif_effects:
        if groups is None:
            raise ValueError("dif_effects requires groups to be provided")
        labels = np.asarray(groups)
        focal = labels == focal_label
        # Per-person beta matrix: base difficulties for everyone, plus the
        # DIF shifts for focal members on the flagged items only.
        beta_pp = np.tile(beta, (n_persons, 1))
        for item_idx, shift in dif_effects.items():
            beta_pp[focal, item_idx] += shift
        probs = expit(theta[:, None] - beta_pp)
    else:
        probs = expit(theta[:, None] - beta[None, :])

    responses = rng.binomial(1, probs)

    # Inject known misfit: random flips break the deterministic relation the
    # Rasch model expects, inflating residual-based fit statistics.
    if misfit_items:
        for item_idx in misfit_items:
            flips = rng.random(n_persons) < misfit_prob
            responses[flips, item_idx] = 1 - responses[flips, item_idx]

    columns = {f"item_{i + 1}": responses[:, i] for i in range(n_items)}
    df = pd.DataFrame({"person_id": [f"p{i + 1}" for i in range(n_persons)], **columns})

    if return_parameters:
        return df, {"theta": theta, "beta": beta}
    return df
