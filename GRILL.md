# Grill Session — 2026-08-27

## Project: rasch-per

### Shortcomings

- AGENT.md references `docs/skills/SPEC.md` as the authoritative source for resolved conventions (Q3 adjustment, ETS delta sign, Ferguson delta), but that file does not exist in the repo. Phase 3 implemented `fit.py` (Yen's Q3) and `dimensionality.py` (PCAR) using the standard published formulas rather than a spec-pinned variant. Risk: if the spec specified a non-standard Q3 adjustment or PCAR centering, current output diverges from intended behavior.

### Unresolved Questions

- Where is the original build spec (the SPEC.md source)? It drove Phases 0-2 but is not on disk. Does the user have it, or was it only in the earlier chat session?
- Should the Q3 / PCAR implementation use the classic residual-correlation form (current) or a spec-specific adjustment once recovered?

### Next Steps

- Recover or re-supply SPEC.md (or the relevant convention snippets) and re-verify fit.py / dimensionality.py against it before Phase 5 (DIF, which also depends on the ETS delta sign convention).

## Resolved 2026-08-27

- The SPEC.md gap (shortcoming above) was closed by reconstructing `docs/skills/SPEC.md` from the implemented code. The original Phase 0-2 spec remains unavailable; the reconstructed conventions use the standard published forms (Yen 1984 Q3, ETS delta sign positive = easier for focal, Ferguson delta per Ferguson 1949). If the original spec is recovered, diff and correct before a public release.
