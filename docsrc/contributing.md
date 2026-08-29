# Contributing

Thanks for your interest in improving `rasch-per`.

## Reporting issues

Open an issue describing the problem, the data shape you used (synthetic or
real), and the command or code that reproduced it. Please do not share
identifiable respondent data.

## Development setup

```bash
git clone https://github.com/aditya-an1l/rasch-per
cd rasch-per
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Validation loop

Run the local validation gate before pushing:

```bash
./.validation.sh
```

It runs the build, type checks (`mypy`), lint (`ruff`), and the test suite with
coverage. New code should keep coverage at or above the project gate.

## Tests

```bash
pytest
```

Tests use synthetic data only. Stretch analyses under `scripts/` (R
cross-validation, CFA, Stocking-Lord linking, PDF export) use the optional
`pdf` and `cfa` extras.

## Documentation

The documentation site lives in `docsrc/` and is built with mkdocs-material.
Preview locally with:

```bash
mkdocs serve
```

See [CONTRIBUTING.md](https://github.com/aditya-an1l/rasch-per/blob/main/CONTRIBUTING.md)
for the full guidelines.
