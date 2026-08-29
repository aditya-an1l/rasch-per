# API Reference

All public symbols are importable from the top-level `rasch_per` package.

## Data

- `ResponseData(matrix, *, strict=True)` - wraps a dichotomous response matrix.
  - `ResponseData.from_csv(path, person_col="person_id")` - load from CSV.
  - `.filter_min_response_rate(threshold)` - drop low-responding persons.
  - `.to_dataframe()`, `.to_numpy()`, `.missing_by_item()`, `.missing_by_person()`.
  - `.n_items`, `.n_persons`, `.item_names`, `.person_ids`.

## Classical Test Theory

- `CTTAnalysis(data, n_boot=1000, seed=None).run() -> CTTResults`
  - `CTTResults.summary()` - DataFrame of item difficulty / discrimination.
  - `CTTResults.reliability` - attribute with `cronbach_alpha`,
    `mcdonald_omega`, `ferguson_delta`.

## Rasch

- `RaschModel().fit(response_data, estimator="MML" | "JML")`
  - `.item_difficulties` - pandas Series indexed by item name.
  - `.person_abilities`, `.responses`, `.item_names`.
  - `.fit_statistics()` - per-item infit / outfit mean-squares.
- `rasch_per.rasch` submodules expose `rasch_probability`, `item_information`,
  `test_information(betas, theta)`, `sem(betas, theta)`,
  `person_separation_reliability`, and `run_pcar(model)`.

## Differential Item Functioning

- `DIFAnalysis(model, groups, reference, focal, estimator="MML", alpha=0.05).analyze() -> DIFResults`
  - `DIFResults.summary()` - per-item Lord chi-square, ETS delta, BH flags.

## Report

- `generate_report(data, output=None, *, groups=None, reference=None, focal=None, estimator="MML", title=...)`
  - Writes a self-contained HTML validity report. `data` is a DataFrame.
