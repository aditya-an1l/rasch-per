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

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2
import matplotlib.pyplot as plt
import pandas as pd

from rasch_per import plots
from rasch_per.ctt import CTTAnalysis
from rasch_per.data import ResponseData
from rasch_per.dif import DIFAnalysis
from rasch_per.dimensionality import run_pcar
from rasch_per.rasch.model import RaschModel

if TYPE_CHECKING:
    from rasch_per.ctt import CTTResults

_TEMPLATE = jinja2.Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: Helvetica, Arial, sans-serif; margin: 2rem; color: #222; }
  h1 { border-bottom: 2px solid #444; }
  h2 { margin-top: 2rem; border-bottom: 1px solid #aaa; }
  table { border-collapse: collapse; margin: 0.5rem 0; }
  th, td { border: 1px solid #ccc; padding: 3px 8px; text-align: right; }
  th { background: #eee; }
  .figure { margin: 1rem 0; }
  .placeholder { color: #777; font-style: italic; }
  img { max-width: 100%; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
{% for section in sections %}
<h2>{{ section.title }}</h2>
{{ section.body }}
{% endfor %}
</body>
</html>
"""
)


def _fig_to_b64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _figure_html(fig: Any, alt: str) -> str:
    return (
        f'<div class="figure"><img alt="{alt}" '
        f'src="data:image/png;base64,{_fig_to_b64(fig)}"></div>'
    )


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(float_format=lambda v: f"{v:.4f}")


def _reliability_table(ctt: CTTResults) -> pd.DataFrame:
    rel = ctt.reliability
    return pd.DataFrame(
        {
            "statistic": ["cronbach_alpha", "mcdonald_omega", "ferguson_delta"],
            "value": [rel.cronbach_alpha, rel.mcdonald_omega, rel.ferguson_delta],
        }
    )


def generate_report(
    data: pd.DataFrame,
    output: Any = None,
    *,
    groups: object = None,
    reference: str | None = None,
    focal: str | None = None,
    estimator: str = "MML",
    title: str = "Rasch/CTT Validity Report",
) -> None:
    """Run the full pipeline and write a self-contained HTML validity report.

    Parameters
    ----------
    data : pandas.DataFrame
        Item-by-person dichotomous response matrix.
    output : path-like, file-like, or None
        Destination for the HTML. If ``None``, writes ``validity_report.html``
        in the current working directory.
    groups, reference, focal : optional
        Group labels (aligned to ``data.index`` order) enabling the DIF and
        group-comparison sections.
    estimator : str, default "MML"
        Rasch estimation method passed to :meth:`RaschModel.fit`.
    title : str
        Report title.
    """
    ctt = CTTAnalysis(ResponseData(data)).run()
    model = RaschModel().fit(ResponseData(data), estimator=estimator)
    fit_stats = model.fit_statistics()
    pcar = run_pcar(model)

    sections: list[dict[str, str]] = []

    sections.append(
        {
            "title": "1. Test Content",
            "body": '<p class="placeholder">User-supplied qualitative evidence '
            "about item alignment to the construct domain goes here.</p>",
        }
    )
    sections.append(
        {
            "title": "2. Response Process",
            "body": '<p class="placeholder">Qualitative evidence about how '
            "respondents engaged with items (think-aloud, cognitive "
            "interview notes) goes here.</p>",
        }
    )

    internal: list[str] = []
    internal.append(
        f"<p>PCAR first-contrast eigenvalue: "
        f"<b>{pcar.first_contrast_eigenvalue:.3f}</b>. "
        f"Second dimension suspected: "
        f"<b>{pcar.second_dimension_suspected}</b> "
        f"(cutoff &gt; 2.0).</p>"
    )
    internal.append(_table_html(ctt.summary()))
    internal.append(_figure_html(plots.plot_item_difficulty_bar(ctt), "Item difficulty"))
    internal.append(_figure_html(plots.plot_item_discrimination_bar(ctt), "Item discrimination"))
    internal.append("<h3>Reliability</h3>")
    internal.append(_table_html(_reliability_table(ctt)))
    internal.append("<h3>Rasch Fit</h3>")
    internal.append(_table_html(fit_stats))
    internal.append(_figure_html(plots.plot_wright_map(model), "Wright map"))
    internal.append(_figure_html(plots.plot_test_information(model), "Test information"))
    for item in model.item_names:
        internal.append(_figure_html(plots.plot_icc(model, item), f"ICC {item}"))
    sections.append({"title": "3. Internal Structure", "body": "\n".join(internal)})

    if groups is not None and reference is not None and focal is not None:
        relation: list[str] = []
        dif = DIFAnalysis(model, groups, reference=reference, focal=focal).analyze()
        relation.append(_table_html(dif.summary()))
        relation.append(_figure_html(plots.plot_dif_contrasts(dif), "DIF contrasts"))
        relation.append(
            _figure_html(
                plots.plot_group_ability_distributions(model, groups),
                "Group ability distributions",
            )
        )
        sections.append({"title": "4. Relations to Other Variables", "body": "\n".join(relation)})

    html = _TEMPLATE.render(title=title, sections=sections)

    if output is None:
        output = Path("validity_report.html")
    if hasattr(output, "write"):
        output.write(html)
    else:
        Path(output).write_text(html, encoding="utf-8")
