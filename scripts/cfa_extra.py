"""Extra dimensionality check: single-factor CFA on the item correlation matrix.

Uses semopy (declared under the `cfa` extra). This is a confirmatory companion
to the PCAR residual check already in rasch-per: a well-fitting unidimensional
Rasch model should show a single strong factor.

Usage:
    python scripts/cfa_extra.py responses.csv
"""

from __future__ import annotations

import argparse
import sys


def cfa_extra(df) -> dict:
    try:
        import semopy
    except ImportError as exc:
        raise RuntimeError(
            'semopy is required. Install with: pip install "rasch-per[cfa]"'
        ) from exc

    corr = df.corr()
    model_str = "trait =~ " + " + ".join(df.columns)
    mod = semopy.Model(model_str)
    mod.fit(corr)
    stats = semopy.calc_stats(mod)
    return {"stats": stats, "model": mod}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single-factor CFA dimensionality check.")
    parser.add_argument("csv", help="Response CSV (persons x items).")
    parser.add_argument("--index-col", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required.", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv, index_col=args.index_col)
    res = cfa_extra(df)
    print(res["stats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
