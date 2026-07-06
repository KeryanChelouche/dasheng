#!/usr/bin/env python3
"""Plot confusion matrix from cached features.

Usage
-----
# List available cached feature files
python scripts/plot_confusion.py --list

# Plot with default probe (linear)
python scripts/plot_confusion.py --model dasheng_base --dataset mad

# Plot with k-NN probe
python scripts/plot_confusion.py --model dasheng_base --dataset mad --probe knn

# Save to a specific path
python scripts/plot_confusion.py --model dasheng_base --dataset glasgow -o my_cm.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.datasets.glasgow import GlasgowDataset
from eval.datasets.esc50 import ESC50Dataset
from eval.datasets.mad import MADDataset
from eval.probes.knn import KNNProbe
from eval.probes.linear import LinearProbe
from eval.reporting import plot_confusion_matrix

FEATURES_DIR = REPO_ROOT / "results" / "features"
PLOTS_DIR = REPO_ROOT / "results" / "plots"

DATASET_REGISTRY = {
    "glasgow": lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow"),
    "esc50":   lambda: ESC50Dataset(REPO_ROOT / "data" / "ESC-50-master"),
    "mad":     lambda: MADDataset(REPO_ROOT / "data" / "MAD"),
}

PROBE_REGISTRY = {
    "knn":    lambda: KNNProbe(k=10, metric="cosine"),
    "linear": lambda: LinearProbe(C=1.0),
}


def list_cached() -> None:
    files = sorted(FEATURES_DIR.glob("*.npz"))
    if not files:
        print("No cached features found.")
        return
    print("Available cached features:")
    for f in files:
        model, dataset = f.stem.split("__", 1)
        d = np.load(f)
        print(f"  --model {model} --dataset {dataset}  ({d['features'].shape[0]} samples, {d['features'].shape[1]}D)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="List cached feature files and exit.")
    p.add_argument("--model", type=str, help="Model name (as in cache filename).")
    p.add_argument("--dataset", type=str, choices=DATASET_REGISTRY, help="Dataset name.")
    p.add_argument("--probe", type=str, choices=PROBE_REGISTRY, default="linear", help="Probe type (default: linear).")
    p.add_argument("-o", "--output", type=str, default=None, help="Output path (default: results/plots/<model>__<dataset>_cm.png).")
    args = p.parse_args()

    if args.list:
        list_cached()
        return

    if not args.model or not args.dataset:
        p.error("--model and --dataset are required (use --list to see available options)")

    cache_path = FEATURES_DIR / f"{args.model}__{args.dataset}.npz"
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        print("Use --list to see available options.")
        sys.exit(1)

    data = np.load(cache_path)
    features, labels = data["features"], data["labels"]

    dataset = DATASET_REGISTRY[args.dataset]()
    probe = PROBE_REGISTRY[args.probe]()

    # Aggregate predictions across all CV folds
    y_true_all = np.empty(len(labels), dtype=np.int64)
    y_pred_all = np.empty(len(labels), dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(dataset.cv_splits()):
        probe.fit(features[train_idx], labels[train_idx])
        y_pred_all[test_idx] = probe.predict(features[test_idx])
        y_true_all[test_idx] = labels[test_idx]

    out_path = Path(args.output) if args.output else PLOTS_DIR / f"{args.model}__{args.dataset}_cm_{args.probe}.png"
    title = f"{args.model} / {args.dataset} ({args.probe})"
    plot_confusion_matrix(y_true_all, y_pred_all, dataset.class_names, title, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
