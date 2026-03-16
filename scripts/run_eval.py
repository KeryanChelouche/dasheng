#!/usr/bin/env python3
"""Evaluate one or more models on one or more datasets.

Examples
--------
# k-NN + linear probe on Glasgow with Dasheng-base
python scripts/run_eval.py --model dasheng_base --dataset glasgow

# Grid: two models × two datasets
python scripts/run_eval.py \\
    --model dasheng_base dasheng_06B \\
    --dataset glasgow esc50 \\
    --probes knn linear

# Force re-extraction (ignore cache)
python scripts/run_eval.py --model dasheng_base --dataset glasgow --no-cache
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.datasets.esc50 import ESC50Dataset
from eval.datasets.glasgow import GlasgowDataset
from eval.evaluation import run_evaluation
from eval.models.dasheng import DashengExtractor
from eval.models.audiomae import AudioMAEExtractor
from eval.models.mae_imagenet import MAEImageNetExtractor
from eval.probes.knn import KNNProbe
from eval.probes.linear import LinearProbe
from eval.reporting import print_results, save_results
from loguru import logger

# ── Registries ────────────────────────────────────────────────────────────────
# Add new models / datasets / probes here without touching the rest of the code.

_AUDIOMAE_CHECKPOINT = REPO_ROOT / "pretrained.pth"

MODEL_REGISTRY = {
    "dasheng_base":  lambda: DashengExtractor(variant="base"),
    "dasheng_06B":   lambda: DashengExtractor(variant="06B"),
    "dasheng_12B":   lambda: DashengExtractor(variant="12B"),
    "audiomae":      lambda: AudioMAEExtractor(path=str(_AUDIOMAE_CHECKPOINT)),
    "mae_imagenet":  lambda: MAEImageNetExtractor(),
}

DATASET_REGISTRY = {
    "glasgow": lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow"),
    "esc50":   lambda: ESC50Dataset(REPO_ROOT / "data" / "ESC-50-master"),
}

PROBE_REGISTRY = {
    "knn":    lambda: KNNProbe(k=10, metric="cosine"),
    "linear": lambda: LinearProbe(C=1.0),
}

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", nargs="+", choices=MODEL_REGISTRY, required=True,
        metavar="MODEL", help=f"One or more of: {list(MODEL_REGISTRY)}",
    )
    p.add_argument(
        "--dataset", nargs="+", choices=DATASET_REGISTRY, required=True,
        metavar="DATASET", help=f"One or more of: {list(DATASET_REGISTRY)}",
    )
    p.add_argument(
        "--probes", nargs="+", choices=PROBE_REGISTRY,
        default=["knn", "linear"],
        metavar="PROBE", help=f"One or more of: {list(PROBE_REGISTRY)}",
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
        "--output-dir",
        default=str(REPO_ROOT / "results" / "metrics"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_path, level="DEBUG")
    logger.info(f"Device: {device}  |  log → {log_path}")

    for model_name in args.model:
        for dataset_name in args.dataset:
            model   = MODEL_REGISTRY[model_name]()
            dataset = DATASET_REGISTRY[dataset_name]()
            probes  = [PROBE_REGISTRY[p]() for p in args.probes]

            result = run_evaluation(
                model=model,
                dataset=dataset,
                probes=probes,
                device=device,
                batch_size=args.batch_size,
                use_cache=not args.no_cache,
            )

            print_results(result)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = output_dir / f"{model_name}__{dataset_name}__{ts}.json"
            save_results(result, out)
            logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
