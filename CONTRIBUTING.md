# Contributing to rasch-per

Thanks for your interest in contributing.

## Development setup

```bash
git clone https://github.com/aditya-an1l/rasch-per
cd rasch-per
uv venv --python 3.13 .venv
uv pip install -e ".[dev]" -p .venv
```

Or with plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running checks

```bash
./.validation.sh          # build + type check + lint + tests
python -m pytest          # tests only (with coverage)
python -m ruff check .    # lint only
python -m mypy            # types only
```

CI runs the same checks on Python 3.10-3.13 and enforces a 85% coverage gate.
Both `ruff check` and `mypy` must pass before any PR is merged.

## Code style expectations

- Full NumPy-style docstrings (Parameters, Returns, worked mini-example) on
  every public function/class.
- Type hints everywhere; no new hard dependencies without discussion first.
- Statistical formulas follow the conventions fixed in the project spec; do
  not swap in alternative conventions from other sources without raising it
  for discussion first.
- All example/test data comes from `rasch_per.simulate` - never vendor
  third-party data or item text.

## Publishing (maintainers)

The GitHub Actions workflow `.github/workflows/publish.yml` builds and
publishes via PyPI Trusted Publishing on tagged releases (`v*.*.*`). TestPyPI
publishes on every push to `main`.

Manual publish (first-time alternative):

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
twine upload --repository testpypi dist/*     # sanity check first
twine upload dist/*                            # real release
```
