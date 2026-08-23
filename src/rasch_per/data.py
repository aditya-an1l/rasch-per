"""Loading and validation of dichotomous response data.

Provides the :class:`ResponseData` container: validates that response values
are in {0, 1, NaN}, reports missingness per item and per person, supports
filtering respondents by minimum response rate, and exposes ``n_items``,
``n_persons``, ``item_names`` and ``to_numpy()``.

Spec reference: section 6.0 of the project build spec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["ResponseData"]

_FALSY_STRINGS = {"false", "f", "no", "n", "0"}
_TRUTHY_STRINGS = {"true", "t", "yes", "y", "1"}


def _coerce_value(value: object, strict: bool) -> float:
    """Coerce a single raw cell to 0.0, 1.0 or NaN.

    Parameters
    ----------
    value : object
        Raw cell value from user data.
    strict : bool
        If True, only {0, 1, NaN} are accepted (ValueError otherwise).
        If False, truthy/falsy values (booleans, "true"/"false" strings,
        non-zero numbers) are coerced to 1.0/0.0 first.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the value cannot be interpreted as a dichotomous response.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "nan", "na", "null"):
            return float("nan")
        if not strict:
            if text in _TRUTHY_STRINGS:
                return 1.0
            if text in _FALSY_STRINGS:
                return 0.0
        raise ValueError(
            f"Cannot interpret string {value!r} as a dichotomous response. "
            "Use ResponseData(..., strict=False) to coerce truthy/falsy strings."
        )
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return float("nan")
        if numeric in (0.0, 1.0):
            return numeric
        if not strict:
            return 1.0 if bool(numeric) else 0.0
        raise ValueError(
            f"Response values must be in {{0, 1, NaN}}, got {numeric!r}. "
            "Use ResponseData(..., strict=False) to coerce truthy/falsy values."
        )
    raise ValueError(f"Unsupported response cell type: {type(value).__name__}")


class ResponseData:
    """A validated container for dichotomous item responses.

    Rows are persons/respondents, columns are items; allowed values are
    {0, 1, NaN} with NaN meaning missing/not administered.

    Parameters
    ----------
    matrix : pandas.DataFrame or array-like
        The person x item response matrix. If a DataFrame with a
        ``person_id`` column is given, that column becomes the index.
    item_names : list of str, optional
        Column names when ``matrix`` is an array-like.
    person_ids : list, optional
        Row labels when ``matrix`` is an array-like.
    strict : bool, default True
        If True (default), raise ValueError on any value outside
        {0, 1, NaN}. If False, coerce truthy/falsy values first.

    Raises
    ------
    ValueError
        On invalid response values (strict mode) or unsupported types.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"q1": [1, 0], "q2": [0, 1]})
    >>> rd = ResponseData(df)
    >>> rd.n_items, rd.n_persons
    (2, 2)
    """

    def __init__(
        self,
        matrix: pd.DataFrame | np.ndarray,
        item_names: list[str] | None = None,
        person_ids: list | None = None,
        *,
        strict: bool = True,
    ) -> None:
        if isinstance(matrix, pd.DataFrame):
            frame = matrix.copy()
            if "person_id" in frame.columns:
                frame = frame.set_index("person_id")
        else:
            arr = np.asarray(matrix)
            columns = (
                item_names
                if item_names is not None
                else [f"item_{i + 1}" for i in range(arr.shape[1])]
            )
            index = person_ids if person_ids is not None else range(arr.shape[0])
            frame = pd.DataFrame(arr, index=index, columns=columns)

        if frame.shape[1] == 0 or frame.shape[0] == 0:
            raise ValueError("Response matrix must have at least one row and one column")

        coerced = pd.DataFrame(
            {col: [_coerce_value(v, strict) for v in frame[col]] for col in frame.columns},
            index=frame.index,
            columns=frame.columns,
        )
        self._df = coerced.astype(float)

    @classmethod
    def from_csv(
        cls, path: str | Path, person_col: str | None = "person_id", **kwargs: object
    ) -> ResponseData:
        """Load responses from a CSV file.

        Parameters
        ----------
        path : str or Path
            Path to the CSV file.
        person_col : str or None, default "person_id"
            Name of an optional person identifier column; used as index when
            present. Pass None to treat every column as an item.
        **kwargs
            Extra keyword arguments forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        ResponseData
        """
        df = pd.read_csv(path)
        if person_col is not None and person_col in df.columns:
            df = df.set_index(person_col)
        return cls(df)

    def missing_by_item(self) -> pd.Series:
        """Percentage of missing responses per item."""
        return self._df.isna().mean() * 100

    def missing_by_person(self) -> pd.Series:
        """Percentage of missing responses per person."""
        return self._df.isna().mean(axis=1) * 100

    def filter_min_response_rate(self, threshold: float) -> ResponseData:
        """Keep only persons whose response rate is at least ``threshold``.

        Parameters
        ----------
        threshold : float
            Minimum fraction of items answered (0.0-1.0). Persons answering
            strictly fewer items than ``threshold * n_items`` are dropped.

        Returns
        -------
        ResponseData
            A new filtered container (the original is unchanged).
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        rate = self._df.notna().mean(axis=1)
        kept = self._df.loc[rate >= threshold]
        clone = self.__class__.__new__(self.__class__)
        clone._df = kept.copy()
        return clone

    @property
    def n_items(self) -> int:
        """Number of items."""
        return int(self._df.shape[1])

    @property
    def n_persons(self) -> int:
        """Number of persons/respondents."""
        return int(self._df.shape[0])

    @property
    def item_names(self) -> list[str]:
        """Item column names."""
        return [str(c) for c in self._df.columns]

    @property
    def person_ids(self) -> list:
        """Person identifiers (the row index)."""
        return list(self._df.index)

    def to_numpy(self) -> np.ndarray:
        """Return the response matrix as a float ndarray (NaN for missing)."""
        return self._df.to_numpy(dtype=float)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the internal validated DataFrame."""
        return self._df.copy()

    def __repr__(self) -> str:
        n_missing = int(self._df.isna().sum().sum())
        return (
            f"ResponseData(n_persons={self.n_persons}, n_items={self.n_items}, "
            f"missing_cells={n_missing})"
        )
