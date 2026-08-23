"""Loading and validation of dichotomous response data.

Provides the :class:`ResponseData` container: validates that response values
are in {0, 1, NaN}, reports missingness per item and per person, supports
filtering respondents by minimum response rate, and exposes ``n_items``,
``n_persons``, ``item_names`` and ``to_numpy()``.

Spec reference: section 6.0 of the project build spec.
"""

from __future__ import annotations

__all__ = ["ResponseData"]


class ResponseData:
    """Validated dichotomous response data container (implemented in Phase 1)."""

    def __init__(self, matrix: object) -> None:
        raise NotImplementedError("ResponseData is implemented in Phase 1")
