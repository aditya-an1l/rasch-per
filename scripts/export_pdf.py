"""Convert a rasch-per HTML report (or any HTML) to PDF using WeasyPrint.

Optional dependency (declared in pyproject under the `pdf` extra):
    pip install "rasch-per[pdf]"

Usage:
    python scripts/export_pdf.py report.html -o report.pdf
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert an HTML report to PDF.")
    parser.add_argument("html", help="Input HTML file.")
    parser.add_argument("-o", "--output", default=None, help="Output PDF path.")
    args = parser.parse_args(argv)

    try:
        from weasyprint import HTML
    except ImportError:
        print(
            'weasyprint is required. Install with: pip install "rasch-per[pdf]"',
            file=sys.stderr,
        )
        return 2

    out = args.output or (args.html.rsplit(".", 1)[0] + ".pdf")
    HTML(filename=args.html).write_pdf(out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
