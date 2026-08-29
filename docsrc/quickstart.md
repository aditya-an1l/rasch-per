# Quickstart

## CLI

```bash
pip install rasch-per
rasch-per simulate --output demo.csv
rasch-per analyze demo.csv --output report.html
```

`simulate` writes a synthetic response CSV (with a `person_id` index). `analyze`
reads it back (first column is the person index), runs the full pipeline, and
writes a self-contained HTML report.

```bash
# With differential item functioning (DIF) by a group column
rasch-per analyze demo.csv --groups groups.csv --dif-group gender \
    --reference Man --focal Non-man --output report.html
```

## Python API

```python
import pandas as pd
from rasch_per import (
    ResponseData,
    CTTAnalysis,
    RaschModel,
    DIFAnalysis,
    generate_report,
)

# Load a response matrix (persons as rows, items as columns)
df = pd.read_csv("responses.csv", index_col=0)
data = ResponseData(df)

# Classical Test Theory
ctt = CTTAnalysis(data).run()
print(ctt.summary())
print(ctt.reliability.cronbach_alpha)  # attribute, not a method

# Rasch (MML is the default estimator)
model = RaschModel().fit(data, estimator="MML")
print(model.item_difficulties)  # pandas Series indexed by item name
print(model.fit_statistics())  # infit / outfit mean-squares

# Differential Item Functioning
groups = pd.read_csv("groups.csv", index_col=0)["gender"].reindex(data.person_ids)
dif = DIFAnalysis(model, groups=groups.to_numpy(), reference="Man", focal="Non-man").analyze()
print(dif.summary())  # ETS delta classification + BH flags

# Self-contained HTML validity report
generate_report(
    df,
    output="validity_report.html",
    groups=groups.to_numpy(),
    reference="Man",
    focal="Non-man",
)
```

### API notes

- `generate_report` takes a `pandas.DataFrame`, not a `ResponseData`. When you
  pass `groups`, they must be aligned to the DataFrame's row order.
- `CTTResults.reliability` is an attribute (`cronbach_alpha`, `mcdonald_omega`,
  `ferguson_delta`), not a callable.
- `DIFAnalysis` is run with `.analyze()` (it returns a `DIFResults`).

## Worked notebooks

- `examples/notebooks/quickstart.ipynb` - core workflow end to end.
- `examples/notebooks/report_walkthrough.ipynb` - DIF analysis and report.
