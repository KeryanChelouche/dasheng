#!/usr/bin/env python3
"""Print a Markdown comparison table of all results in results/metrics/.

Usage
-----
python scripts/compare_results.py
python scripts/compare_results.py --dir results/metrics
python scripts/compare_results.py --dataset glasgow
python scripts/compare_results.py --dataset glasgow esc50
python scripts/compare_results.py --model dasheng_base
python scripts/compare_results.py --model dasheng_base audiomae
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.reporting import make_comparison_table, make_cross_comparison_table, make_few_shot_table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        default=str(REPO_ROOT / "results" / "metrics"),
        help="Directory containing .json result files.",
    )
    p.add_argument(
        "--cross-dir",
        default=str(REPO_ROOT / "results" / "cross_eval"),
        help="Directory containing cross-eval .json result files.",
    )
    p.add_argument(
        "--few-shot-dir",
        nargs="+",
        default=[
            str(REPO_ROOT / "results" / "few_shot"),
            str(REPO_ROOT / "results" / "cross_few_shot"),
        ],
        help="Directories containing few-shot .json result files.",
    )
    p.add_argument(
        "--few-shot-probe",
        default="knn_k10",
        choices=["knn_k10", "linear_C1.0", "supervised"],
        help="Probe to show in few-shot table (default: knn_k10).",
    )
    p.add_argument(
        "--few-shot-metric",
        default="acc",
        choices=["acc", "f1"],
        help="Metric to show in few-shot table (default: acc).",
    )
    p.add_argument(
        "--dataset",
        nargs="+",
        metavar="DATASET",
        default=None,
        help="Only show results for these dataset(s). Omit to show all.",
    )
    p.add_argument(
        "--model",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Only show results for these model(s). Omit to show all.",
    )
    args = p.parse_args()

    table = make_comparison_table(Path(args.dir), datasets=args.dataset, models=args.model)
    print(table)

    cross_dir = Path(args.cross_dir)
    if cross_dir.is_dir() and any(cross_dir.glob("*.json")):
        print("\n## Cross-dataset evaluation\n")
        cross_table = make_cross_comparison_table(
            cross_dir, models=args.model, datasets=args.dataset
        )
        print(cross_table)

    few_shot_dirs = [Path(d) for d in args.few_shot_dir]
    if any(d.is_dir() and any(d.glob("*.json")) for d in few_shot_dirs):
        print(f"\n## Few-shot evaluation ({args.few_shot_probe}, {args.few_shot_metric})\n")
        fs_table = make_few_shot_table(
            few_shot_dirs,
            probe=args.few_shot_probe,
            metric=args.few_shot_metric,
            datasets=args.dataset,
            models=args.model,
        )
        print(fs_table)


if __name__ == "__main__":
    main()
