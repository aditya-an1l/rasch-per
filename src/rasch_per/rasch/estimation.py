"""Parameter estimation for the Rasch model.

Two estimators:

- **JML** (Joint Maximum Likelihood): alternating conditional MLE for theta
  and beta until convergence; identifiability constraint mean(beta) = 0;
  extreme scores handled with the standard adjustment (see below).
- **MML** (Marginal Maximum Likelihood, default): theta ~ N(0, sigma^2),
  integrated out via Gauss-Hermite quadrature; item parameters estimated by
  EM. Mirrors R's TAM default.

Extreme-score convention (resolved per spec section 6.2): persons or items
with all-correct / all-incorrect observed responses have infinite MLEs, so
their effective score is pulled half a point from the boundary
(perfect: n_obs - 0.5; zero: 0.5). This yields finite estimates shrunk at
least ~0.3 logits from the asymptote - the standard "extreme measures"
adjustment - and keeps those respondents in the calibration instead of
dropping them.

Spec reference: section 6.2 (estimation) of the project build spec.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

__all__ = ["EstimationResult", "fit_jml", "fit_mml"]

_EPS = 1e-10


@dataclass(frozen=True)
class EstimationResult:
    """Estimated Rasch parameters and standard errors.

    Attributes
    ----------
    betas : ndarray, shape (n_items,)
        Item difficulties, constrained to mean 0.
    se_beta : ndarray, shape (n_items,)
    thetas : ndarray, shape (n_persons,)
        Person abilities.
    se_theta : ndarray, shape (n_persons,)
    converged : bool
        Whether the outer iteration loop met ``tol`` within ``max_iter``.
    """

    betas: np.ndarray
    se_beta: np.ndarray
    thetas: np.ndarray
    se_theta: np.ndarray
    converged: bool


def _observed_mask(matrix: np.ndarray) -> np.ndarray:
    return ~np.isnan(matrix)


def _initial_betas(matrix: np.ndarray) -> np.ndarray:
    """Logit-transformed item p-values, centered to mean 0."""
    counts = _observed_mask(matrix).sum(axis=0)
    scores = np.nansum(matrix, axis=0)
    p = np.clip(scores / counts, _EPS, 1 - _EPS)
    betas = -np.log(p / (1 - p))  # higher p -> easier -> lower beta
    return betas - betas.mean()


def _initial_thetas(matrix: np.ndarray) -> np.ndarray:
    """Logit-transformed person raw scores, centered to mean 0."""
    n_obs = _observed_mask(matrix).sum(axis=1)
    scores = np.nansum(matrix, axis=1)
    safe_n = np.maximum(n_obs, 1)
    p = np.clip(scores / safe_n, _EPS, 1 - _EPS)
    thetas = np.log(p / (1 - p))
    return thetas - thetas.mean()


def _adjust_extreme_scores(scores: np.ndarray, n_obs: np.ndarray) -> tuple[np.ndarray, bool]:
    """Pull all-correct / all-incorrect effective scores half a point in."""
    adjusted = scores.astype(float).copy()
    extreme = (scores <= 0) | (scores >= n_obs)
    adjusted[scores >= n_obs] = n_obs[scores >= n_obs] - 0.5
    adjusted[scores <= 0] = 0.5
    return adjusted, bool(extreme.any())


def _newton_theta_given_beta(
    matrix: np.ndarray,
    mask: np.ndarray,
    betas: np.ndarray,
    start: np.ndarray,
    max_inner: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Person MLEs given fixed betas; returns (thetas, se_thetas).

    Newton-Raphson on each person's conditional log-likelihood. Extreme raw
    scores use the half-point-adjusted score so their estimate stays finite.
    The SE is 1/sqrt(test information at theta).
    """
    n_obs = mask.sum(axis=1)
    scores = np.where(mask, matrix, 0.0).sum(axis=1)
    eff_scores, had_extreme = _adjust_extreme_scores(scores, n_obs)
    if had_extreme:
        warnings.warn(
            "Extreme scores present (all-correct/all-incorrect); estimates for "
            "those persons use the standard half-point score adjustment.",
            stacklevel=3,
        )

    thetas = start.copy()
    for _ in range(max_inner):
        d = thetas[:, None] - betas[None, :]
        p = _expit_masked(d, mask)
        info = (p * (1 - p)).sum(axis=1)
        grad = eff_scores - p.sum(axis=1)
        step = np.divide(grad, info, out=np.zeros_like(grad), where=info > _EPS)
        thetas += step
        if np.max(np.abs(step)) < _EPS:
            break

    d = thetas[:, None] - betas[None, :]
    p = _expit_masked(d, mask)
    info = (p * (1 - p)).sum(axis=1)
    se = np.divide(1.0, np.sqrt(info), out=np.full_like(info, np.nan), where=info > _EPS)
    return thetas, se


def _expit_masked(d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sigmoid of ``d`` with unobserved entries zeroed out of sums."""
    return np.where(mask, expit(d), 0.0)


def fit_jml(matrix: np.ndarray, max_iter: int = 500, tol: float = 1e-6) -> EstimationResult:
    """Joint maximum likelihood estimation.

    Alternates conditional MLE updates of theta (given beta) and beta (given
    theta) until parameter change falls below ``tol``. Identifiability is
    imposed by constraining mean(beta) = 0 after every update.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (n_persons, n_items)
        Dichotomous response matrix with NaN for missing.
    max_iter : int
    tol : float

    Returns
    -------
    EstimationResult
    """
    arr = np.asarray(matrix, dtype=float)
    mask = _observed_mask(arr)

    # Items answered by nobody (or with no variation) cannot be calibrated.
    item_counts = mask.sum(axis=0)
    if np.any(item_counts == 0):
        raise ValueError("Cannot calibrate items that were never administered")

    betas = _initial_betas(arr)
    thetas = _initial_thetas(arr)
    converged = False
    for _ in range(max_iter):
        thetas, _ = _newton_theta_given_beta(arr, mask, betas, thetas)
        new_betas = _newton_beta_given_theta(arr, mask, thetas, betas)
        shift = np.max(np.abs(new_betas - betas))
        betas = new_betas
        if shift < tol:
            converged = True
            break

    thetas, se_theta = _newton_theta_given_beta(arr, mask, betas, thetas)
    se_beta = _beta_se(arr, mask, thetas)
    return EstimationResult(
        betas=betas,
        se_beta=se_beta,
        thetas=thetas,
        se_theta=se_theta,
        converged=converged,
    )


def _newton_beta_given_theta(
    arr: np.ndarray, mask: np.ndarray, thetas: np.ndarray, start: np.ndarray
) -> np.ndarray:
    """One full Newton solve of beta given theta, then re-centered."""
    scores = np.where(mask, arr, 0.0).sum(axis=0)
    n_obs = mask.sum(axis=0)
    eff_scores, _ = _adjust_extreme_scores(scores, n_obs)
    betas = start.copy()
    for _ in range(100):
        d = thetas[:, None] - betas[None, :]
        p = _expit_masked(d, mask)
        info = (p * (1 - p)).sum(axis=0)
        grad = p.sum(axis=0) - eff_scores
        step = np.divide(grad, info, out=np.zeros_like(grad), where=info > _EPS)
        betas += step
        if np.max(np.abs(step)) < _EPS:
            break
    return betas - betas.mean()


def _beta_se(arr: np.ndarray, mask: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    d = thetas[:, None] - arr  # NaN entries produce NaN p; excluded below
    p = expit(d)
    info = np.nansum(p * (1 - p), axis=0)
    return np.divide(1.0, np.sqrt(info), out=np.full(arr.shape[1], np.nan), where=info > _EPS)


def fit_mml(
    matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    n_nodes: int = 61,
    sigma: float = 1.0,
) -> EstimationResult:
    """Marginal maximum likelihood estimation via Gauss-Hermite EM.

    Assumes theta ~ N(0, sigma^2) with sigma fixed (default 1.0); the
    population ability distribution is integrated numerically with
    ``n_nodes`` Gauss-Hermite nodes. Item difficulties are updated by EM:
    the E-step computes posterior node weights per person, the M-step takes a
    Newton step on the expected complete-data log-likelihood per item.

    Person abilities are reported as EAP (expected a posteriori) estimates
    with posterior-standard-deviation SEs.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (n_persons, n_items)
        Dichotomous response matrix with NaN for missing.
    max_iter : int
    tol : float
        Convergence tolerance on the maximum beta change between iterations.
    n_nodes : int
        Number of quadrature nodes.
    sigma : float
        Fixed population SD identifying the scale.

    Returns
    -------
    EstimationResult
    """
    from numpy.polynomial.hermite import hermgauss

    arr = np.asarray(matrix, dtype=float)
    mask = _observed_mask(arr)
    if np.any(mask.sum(axis=0) == 0):
        raise ValueError("Cannot calibrate items that were never administered")

    nodes, weights = hermgauss(n_nodes)
    theta_q = np.sqrt(2.0) * sigma * nodes
    log_w = np.log(weights) - np.log(weights.sum())

    betas = _initial_betas(arr)
    x = np.where(mask, arr, 0.0)
    obs_counts = x.sum(axis=0)  # constant across EM iterations

    def posteriors(current_betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """E-step: return (h, p1) with h of shape (Q, N), p1 of shape (Q, N, J).

        ``h[q, n]`` is the posterior probability that person n sits at node q;
        ``p1[q, n, j]`` is P(correct | node q, item j), zeroed where missing.
        """
        d_qj = theta_q[:, None] - current_betas[None, :]  # (Q, J)
        # log P(x=1) = -logaddexp(0, beta - theta); log P(x=0) symmetric.
        log_p1 = -np.logaddexp(0.0, -d_qj)
        log_p0 = -np.logaddexp(0.0, d_qj)
        ll_terms = x[None, :, :] * log_p1[:, None, :] + (1 - x[None, :, :]) * log_p0[:, None, :]
        ll = np.where(mask[None, :, :], ll_terms, 0.0)  # (Q, N, J)
        log_h = ll.sum(axis=2) + log_w[:, None]
        log_h -= log_h.max(axis=0, keepdims=True)
        h_out = np.exp(log_h)
        h_out /= h_out.sum(axis=0, keepdims=True)
        p1_out = np.where(mask[None, :, :], expit(d_qj)[:, None, :], 0.0)
        return h_out, p1_out

    converged = False
    for _ in range(max_iter):
        h, p1 = posteriors(betas)

        # M-step: one Newton step per item on the expected log-likelihood.
        # Gradient wrt beta_j is (predicted - observed correct counts);
        # with Hessian -info, the Newton step is beta += (pred - obs)/info.
        expected = np.einsum("qn,qnj->j", h, p1)
        info = np.einsum("qn,qnj->j", h, p1 * (1 - p1))
        new_betas = betas + np.divide(
            expected - obs_counts,
            info,
            out=np.zeros_like(betas),
            where=(info > _EPS),
        )
        shift = np.max(np.abs(new_betas - betas))
        betas = new_betas
        if shift < tol:
            converged = True
            break

    # Final E-step for person EAP estimates.
    h, p1 = posteriors(betas)

    post_mean = (theta_q[:, None] * h).sum(axis=0)
    post_var = (theta_q[:, None] ** 2 * h).sum(axis=0) - post_mean**2

    # Beta SEs from the expected information at the posterior node weights.
    info_beta = np.einsum("qn,qnj->j", h, p1 * (1 - p1))
    se_beta = 1.0 / np.sqrt(info_beta)

    # Persons who answered nothing get NaN estimates.
    empty = mask.sum(axis=1) == 0
    post_mean = np.where(empty, np.nan, post_mean)
    post_var = np.where(empty, np.nan, post_var)

    return EstimationResult(
        betas=betas,
        se_beta=se_beta,
        thetas=post_mean,
        se_theta=np.sqrt(np.maximum(post_var, 0.0)),
        converged=converged,
    )
