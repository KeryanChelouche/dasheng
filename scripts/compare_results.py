#!/usr/bin/env python3
"""Print a Markdown comparison table of all results in results/metrics/.

Usage
-----
python scripts/compare_results.py
python scripts/compare_results.py --dir results/metrics
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.reporting import make_comparison_table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        default=str(REPO_ROOT / "results" / "metrics"),
        help="Directory containing .json result files.",
    )
    args = p.parse_args()

    table = make_comparison_table(Path(args.dir))
    print(table)


if __name__ == "__main__":
    main()
