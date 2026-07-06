#!/usr/bin/env python3
"""Compute mutual information between extracted features and labels.

Uses sklearn's mutual_info_classif (k-NN based MI estimation for
continuous features vs discrete labels).

Examples
--------
# Single model, single dataset
python scripts/run_mi.py --model dasheng_base --dataset glasgow

# Compare MI across multiple models
python scripts/run_mi.py --model dasheng_base audiomae beats_iter3 --dataset glasgow

# Custom number of neighbors for MI estimation
python scripts/run_mi.py --model dasheng_base --dataset glasgow --n-neighbors 5
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.features import extract_and_cache
from loguru import logger

# Reuse registries from run_eval
from scripts.run_eval import DATASET_REGISTRY, MODEL_REGISTRY


def compute_mi(
    features: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> dict:
    """Compute mutual information between features and labels.

    Returns a dict with:
        - mi_per_feature: MI for each feature dimension
        - mi_sum: total MI (sum across features)
        - mi_mean: average MI per feature
        - mi_median: median MI per feature
        - mi_max: maximum MI across features
        - n_features: number of feature dimensions
    """
    # Standardise so the k-NN distances are comparable across dimensions
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    mi = mutual_info_classif(
        X, labels,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    return {
        "mi_per_feature": mi.tolist(),
        "mi_sum": float(mi.sum()),
        "mi_mean": float(mi.mean()),
        "mi_median": float(np.median(mi)),
        "mi_max": float(mi.max()),
        "mi_std": float(mi.std()),
        "n_features": int(len(mi)),
        "n_nonzero": int((mi > 0).sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", nargs="+", choices=list(MODEL_REGISTRY), required=True,
        metavar="MODEL", help=f"One or more of: {list(MODEL_REGISTRY)}",
    )
    p.add_argument(
        "--dataset", nargs="+", choices=list(DATASET_REGISTRY), required=True,
        metavar="DATASET", help=f"One or more of: {list(DATASET_REGISTRY)}",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--no-cache", action="store_true",
        help="Re-extract features even if a cache file exists.",
    )
    p.add_argument(
        "--n-neighbors", type=int, default=3,
        help="Number of neighbors for MI estimation (default: 3).",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "mi"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"mi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_path, level="DEBUG")
    logger.info(f"Device: {device}  |  log → {log_path}")

    all_results = []

    for model_name in args.model:
        model = MODEL_REGISTRY[model_name]()

        for dataset_name in args.dataset:
            dataset = DATASET_REGISTRY[dataset_name]()

            model.to(device)
            features, labels = extract_and_cache(
                model, dataset, device,
                batch_size=args.batch_size,
                use_cache=not args.no_cache,
            )

            logger.info(
                f"Computing MI — model={model_name}  dataset={dataset_name}  "
                f"features={features.shape}  n_neighbors={args.n_neighbors}"
            )

            mi_result = compute_mi(
                features, labels,
                n_neighbors=args.n_neighbors,
            )

            result = {
                "model": model_name,
                "dataset": dataset_name,
                "n_samples": int(len(labels)),
                "n_classes": dataset.n_classes,
                "n_neighbors": args.n_neighbors,
                **{k: v for k, v in mi_result.items() if k != "mi_per_feature"},
            }
            all_results.append(result)

            # Print summary
            print(f"\n{'='*60}")
            print(f"Model:   {model_name}")
            print(f"Dataset: {dataset_name}  ({len(labels)} samples, {dataset.n_classes} classes)")
            print(f"Features: {features.shape[1]}D")
            print(f"{'-'*60}")
            print(f"  MI sum:       {mi_result['mi_sum']:.4f} nats")
            print(f"  MI mean:      {mi_result['mi_mean']:.4f} nats")
            print(f"  MI median:    {mi_result['mi_median']:.4f} nats")
            print(f"  MI max:       {mi_result['mi_max']:.4f} nats")
            print(f"  MI std:       {mi_result['mi_std']:.4f} nats")
            print(f"  Non-zero MI:  {mi_result['n_nonzero']}/{mi_result['n_features']} features")
            print(f"{'='*60}")

            # Save full result (including per-feature MI) to JSON
            full_result = {**result, "mi_per_feature": mi_result["mi_per_feature"]}
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = output_dir / f"mi__{model_name}__{dataset_name}__{ts}.json"
            out.write_text(json.dumps(full_result, indent=2))
            logger.info(f"Saved → {out}")

    # Print comparison table if multiple runs
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("Comparison")
        print(f"{'='*80}")
        header = f"  {'model':<25} {'dataset':<15} {'MI sum':>10} {'MI mean':>10} {'non-zero':>10}"
        print(header)
        print(f"  {'-'*70}")
        for r in all_results:
            print(
                f"  {r['model']:<25} {r['dataset']:<15} "
                f"{r['mi_sum']:>10.4f} {r['mi_mean']:>10.4f} "
                f"{r['n_nonzero']:>5}/{r['n_features']}"
            )
        print()


if __name__ == "__main__":
    main()
