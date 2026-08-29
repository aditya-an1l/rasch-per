# Command-Line Reference

Install the package first:

```bash
pip install rasch-per
```

The CLI is a thin wrapper over the importable Python API. Every command maps to
a documented library function.

```bash
rasch-per --help
```

## `rasch-per simulate`

Generate a synthetic dichotomous response CSV for trying out the tool.

| Option | Default | Description |
|--------|---------|-------------|
| `--n-persons` | `300` | Number of simulated respondents. |
| `--n-items` | `20` | Number of simulated items. |
| `--seed` | `None` | Random seed for reproducibility. |
| `--output` | `demo.csv` | Output CSV path. |

The output has a `person_id` column plus one 0/1 column per item.

```bash
rasch-per simulate --n-persons 500 --n-items 20 --seed 42 --output demo.csv
```

## `rasch-per analyze`

Run the full validity analysis pipeline and write a self-contained HTML report.

| Option | Default | Description |
|--------|---------|-------------|
| `CSV` (argument) | - | Path to the response CSV (first column is the person index). |
| `--output` | `report.html` | Output report path. |
| `--format` | `html` | Output format (only `html` in this release). |
| `--estimator` | `MML` | Rasch estimation method: `MML` or `JML`. |
| `--groups` | `None` | CSV of `person_id` to group metadata. |
| `--dif-group` | `None` | Column in `--groups` used for DIF. |
| `--reference` | `None` | Reference group label for DIF. |
| `--focal` | `None` | Focal group label for DIF. |
| `--min-response-rate` | `0.5` | Drop respondents answering fewer items than this fraction. |
| `--verbose/--quiet` | verbose | Print progress to stderr. |

`--groups` and `--dif-group` must be supplied together; `--reference` and
`--focal` must also be supplied together. If the group labels are omitted, the
first two distinct labels are used.

```bash
rasch-per analyze demo.csv --output report.html
rasch-per analyze demo.csv --groups groups.csv --dif-group gender \
    --reference Man --focal Non-man --output report.html
```

## `rasch-per validate`

Run data validation and diagnostics without the full analysis. Reports the
number of respondents and items, per-item missingness percentages, and the
response value range (expected 0/1 with possible NaN for missing).
