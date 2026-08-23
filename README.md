# rasch-per

Rasch model and Classical Test Theory (CTT) psychometric analysis for
education research - built for physics/STEM/discipline-based education
researchers (PER/DBER).

Feed it a CSV of dichotomous (0/1) item responses and get person abilities,
item difficulties, fit statistics, dimensionality checks, DIF analysis, and a
full self-contained HTML validity report.

> Status: early scaffolding. The API below is the target design and is being
> implemented phase by phase.

## Quickstart (target)

```bash
pip install rasch-per
rasch-per simulate --output demo.csv
rasch-per analyze demo.csv --output report.html
```

## Python API (target)

```python
import pandas as pd
from rasch_per import ResponseData, CTTAnalysis, RaschModel, DIFAnalysis, generate_report

data = ResponseData.from_csv("responses.csv")

ctt = CTTAnalysis(data).run()
print(ctt.summary())

rasch = RaschModel().fit(data, estimator="MML")
print(rasch.item_difficulties)
print(rasch.fit_statistics())

groups = pd.read_csv("groups.csv")  # person_id, gender
dif = DIFAnalysis(rasch, groups=groups["gender"], reference="Man", focal="Non-man").run()
print(dif.summary())

generate_report(data, output="validity_report.html")
```

## What's inside

- **CTT**: item difficulty (p-values), discrimination (corrected item-total,
  rest-score based) with bootstrap SEs; Cronbach's alpha, McDonald's omega,
  Ferguson's delta.
- **Rasch**: JML and MML estimation (MML default), standard errors.
- **Fit**: infit/outfit mean-square with low-stakes/high-stakes presets;
  Yen's Q3 local independence check.
- **Dimensionality**: PCAR first-contrast eigenvalue diagnostic.
- **DIF**: Lord's chi-square, mean/mean linking, ETS delta effect sizes,
  Benjamini-Hochberg correction.
- **Report**: single-file HTML validity report with embedded plots.

## Methodology & References

This package implements standard, published psychometric methods used
throughout physics/discipline-based education research:

- Rasch model (Rasch, 1960; Wright & Stone, 1979)
- Joint and marginal maximum likelihood estimation (e.g., as in R packages
  `TAM`, `eRm`)
- Infit/outfit mean-square fit statistics (Smith, 2000; Linacre, 2002)
- Yen's Q3 local independence statistic (Yen, 1984)
- Principal components analysis of residuals (Linacre, 1998)
- Lord's chi-square DIF test (Lord, 1980)
- ETS delta scale DIF classification (ETS categories A/B/C)
- Benjamini-Hochberg false discovery rate control (Benjamini & Hochberg, 1995)
- Cronbach's alpha (Cronbach, 1951), McDonald's omega (McDonald, 1999),
  Ferguson's delta (Ferguson, 1949)

No third-party assessment content or data is included; all examples use
synthetic data from the package's own simulator.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Validation loop: `./.validation.sh`.

## License

MIT
