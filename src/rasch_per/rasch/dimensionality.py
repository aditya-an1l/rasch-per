"""Dimensionality diagnostics: Principal Components Analysis of Residuals.

After fitting the Rasch model, standardized residuals are extracted and PCA
is run on them; the eigenvalue of the first contrast is reported. An
eigenvalue above 2.0 suggests a possible second dimension (advisory flag,
never an automatic failure).

Method: standardized residuals ``z_ni`` (person x item) are correlated across
persons to form the residual correlation matrix; its largest eigenvalue is the
"first contrast" eigenvalue. (Per the project AGENT.md a SPEC.md would pin a
specific PCAR/Q3 adjustment variant; that source is currently absent, so the
standard residual-correlation PCA form is used. Flagging for review.)

Optional single-factor CFA via semopy is an extra dependency
(``pip install rasch-per[cfa]``), not part of core v1.

Spec reference: section 6.3 of the project build spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rasch_per.rasch.fit import standardized_residuals

if TYPE_CHECKING:
    from rasch_per.rasch.model import RaschModel

__all__ = ["run_pcar", "PCARResult"]


@dataclass(frozen=True)
class PCARResult:
    """Result of PCA of residuals.

    Attributes
    ----------
    first_contrast_eigenvalue : float
        Largest eigenvalue of the residual correlation matrix. The common
        screening rule flags a possible second dimension when this exceeds 2.0.
    eigenvalues : numpy.ndarray
        All eigenvalues in descending order.
    second_dimension_suspected : bool
        True when ``first_contrast_eigenvalue`` > 2.0 (advisory only).
    """

    first_contrast_eigenvalue: float
    eigenvalues: np.ndarray
    second_dimension_suspected: bool


def run_pcar(model: RaschModel) -> PCARResult:
    """Principal Components Analysis of Rasch residuals.

    Computes the correlation matrix of the standardized residuals across
    persons (pairwise deletion of missing responses) and reports the largest
    eigenvalue as the first-contrast eigenvalue.

    Parameters
    ----------
    model : fitted RaschModel
        Model exposing ``item_difficulties``, ``person_abilities`` and
        ``responses``.

    Returns
    -------
    PCARResult
        Dataclass with the first-contrast eigenvalue, all eigenvalues, and an
        advisory ``second_dimension_suspected`` flag (> 2.0).

    Examples
    --------
    >>> from rasch_per.data import ResponseData
    >>> from rasch_per.rasch import RaschModel
    >>> df = ResponseData([[1, 0, 1], [0, 1, 1]])
    >>> m = RaschModel().fit(df, estimator="JML")
    >>> res = run_pcar(m)
    >>> res.first_contrast_eigenvalue >= 0.0
    True
    """
    z = standardized_residuals(model)
    frame = pd.DataFrame(z, columns=pd.Index(_item_names(model)))
    corr = frame.corr()
    cov = np.asarray(corr.values, dtype=float)
    # Correlation matrices are symmetric PSD; eigh returns ascending eigenvalues.
    eigenvalues = np.linalg.eigh(cov)[0]
    eigenvalues = np.sort(eigenvalues)[::-1]
    first = float(eigenvalues[0])
    return PCARResult(
        first_contrast_eigenvalue=first,
        eigenvalues=eigenvalues,
        second_dimension_suspected=first > 2.0,
    )


def _item_names(model: RaschModel) -> list[str]:
    try:
        return list(model.item_names)
    except AttributeError as exc:  # pragma: no cover
        raise TypeError("run_pcar expects a fitted RaschModel with item_names") from exc
