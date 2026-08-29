# The Validity Report

`generate_report` (and `rasch-per analyze`) produces a single, self-contained
HTML file. All plots are embedded as base64 PNGs, so the file renders fully
offline with no external dependencies. The report is organized as validity
evidence following a standard validity framework:

## 1. Test Content

A placeholder section for user-supplied qualitative evidence about how items
align with the construct domain. Fill this in with your own documentation.

## 2. Response Process

A placeholder section for qualitative evidence about how respondents engaged
with the items (think-aloud studies, cognitive interview notes).

## 3. Internal Structure

The quantitative core of the report:

- **Dimensionality** - PCAR first-contrast eigenvalue and a flag for a
  suspected second dimension (cutoff > 2.0).
- **CTT item table** - difficulty, difficulty SE, discrimination,
  discrimination SE per item.
- **Item difficulty and discrimination bar charts** with bootstrap SEs.
- **Reliability** - Cronbach's alpha, McDonald's omega, Ferguson's delta.
- **Rasch fit** - per-item infit and outfit mean-squares.
- **Wright map** - person ability histogram with item-difficulty markers.
- **Test information and SEM** curve.
- **Item characteristic curves (ICC)** - one plot per item, with an empirical
  overlay.

## 4. Relations to Other Variables

Included only when group labels are supplied. Contains:

- **DIF table** - per-item Lord's chi-square, p-value, Benjamini-Hochberg
  adjusted q-value, ETS delta, and ETS A/B/C classification.
- **DIF contrasts plot**.
- **Group ability distributions plot**.

See [Methodology](methodology.md) for the formulas behind each section.
