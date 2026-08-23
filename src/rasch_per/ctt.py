"""Classical Test Theory (CTT) statistics.

Per item: difficulty (proportion correct) and discrimination (point-biserial
correlation between the item score and the REST score, i.e. total excluding
that item), each with bootstrap standard errors.

Test level: Cronbach's alpha, McDonald's omega, Ferguson's delta, returned in
a :class:`CTTResults` dataclass with ``.summary()`` and a ``.reliability``
sub-object.

Spec reference: section 6.1 of the project build spec.
"""

from __future__ import annotations

__all__ = ["CTTAnalysis", "CTTResults"]


class CTTResults:
    """CTT results container (implemented in Phase 1)."""

    def __init__(self) -> None:
        raise NotImplementedError("CTTResults is implemented in Phase 1")


class CTTAnalysis:
    """CTT analysis runner (implemented in Phase 1)."""

    def __init__(self, data: object) -> None:
        raise NotImplementedError("CTTAnalysis is implemented in Phase 1")
