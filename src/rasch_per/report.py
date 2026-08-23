"""Full-pipeline validity report generation.

:func:`generate_report` runs the entire analysis pipeline end to end and
writes a single self-contained HTML file (plots embedded as base64 PNGs),
organized as validity-evidence sections:

1. Test Content (user-supplied qualitative placeholder)
2. Response Process (qualitative placeholder)
3. Internal Structure (dimensionality, CTT, reliability, Rasch fit,
   Wright map, test information/SEM, DIF)
4. Relations to Other Variables (group comparison when groups are supplied)

Spec reference: section 6.6 of the project build spec.
"""

from __future__ import annotations

__all__ = ["generate_report"]


def generate_report(data: object, output: object = None, **kwargs: object) -> None:
    """Full-pipeline HTML/PDF validity report (implemented in Phase 5)."""
    raise NotImplementedError("generate_report is implemented in Phase 5")
