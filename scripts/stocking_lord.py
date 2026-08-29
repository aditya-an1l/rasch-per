"""Stocking-Lord (Haebara) linking between two Rasch calibrations.

Given item difficulties from two independent Rasch calibrations (e.g. two
groups, or a new form vs an anchor form), find the constant shift that aligns
the two test characteristic curves. In the Rasch model all slopes are 1, so the
linking constant is the intercept difference.

Usage:
    python scripts/stocking_lord.py ref_betas.csv foc_betas.csv -o linking.csv

Each CSV should contain a single column (or a single row) of item difficulties
in the same item order.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize_scalar

try:
    from numpy import trapezoid as _trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapz


def _load_betas(path: str) -> np.ndarray:
    arr = np.asarray(np.genfromtxt(path, delimiter=","), dtype=np.float64)
    arr = np.atleast_1d(arr)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


def stocking_lord(
    b_ref: np.ndarray,
    b_foc: np.ndarray,
    theta_min: float = -4.0,
    theta_max: float = 4.0,
    n: int = 201,
) -> float:
    """Return the shift to apply to focal difficulties to match the reference."""
    thetas = np.linspace(theta_min, theta_max, n)
    ref = 1.0 / (1.0 + np.exp(-(thetas[:, None] - b_ref[None, :]))).sum(axis=1)

    def loss(c: float) -> float:
        foc = 1.0 / (1.0 + np.exp(-((thetas - c)[:, None] - b_foc[None, :]))).sum(axis=1)
        return float(_trapz((ref - foc) ** 2, thetas))

    res = minimize_scalar(loss, bounds=(-2.0, 2.0), method="bounded")
    return float(res.x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stocking-Lord linking of two Rasch calibrations.")
    parser.add_argument("ref", help="Reference item-difficulty CSV.")
    parser.add_argument("foc", help="Focal item-difficulty CSV.")
    parser.add_argument("-o", "--output", default=None, help="Write aligned focal betas here.")
    args = parser.parse_args(argv)

    b_ref = _load_betas(args.ref)
    b_foc = _load_betas(args.foc)
    shift = stocking_lord(b_ref, b_foc)
    print(f"Linking constant (apply to focal betas): {shift:.4f}")
    if args.output:
        aligned = b_foc + shift
        np.savetxt(args.output, aligned, delimiter=",")
        print(f"Wrote aligned focal betas to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
