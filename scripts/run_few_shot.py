#!/usr/bin/env python3
"""Evaluate sample efficiency: fit probes on N-shot subsets of the training data.

Examples
--------
# All models on Glasgow with defaults
python scripts/run_few_shot.py --model dasheng_base --dataset glasgow

# Grid: two models, custom shot levels
python scripts/run_few_shot.py \\
    --model dasheng_base mae_imagenet \\
    --dataset glasgow \\
    --n-shots 1 5 10 50 100

# Force re-extraction (ignore feature cache)
python scripts/run_few_shot.py --model dasheng_base --dataset glasgow --no-cache
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
from eval.few_shot import run_few_shot_evaluation
from eval.models.audiomae import AudioMAEExtractor
from eval.models.beats import BEATsExtractor
from eval.models.dasheng import DashengExtractor
from eval.models.fisher import FISHERExtractor
from eval.models.mae_imagenet import MAEImageNetExtractor
from eval.probes.knn import KNNProbe
from eval.probes.linear import LinearProbe
from loguru import logger

# ── Registries (keep in sync with run_eval.py) ────────────────────────────────

_AUDIOMAE_CHECKPOINT     = REPO_ROOT / "pretrained.pth"
_BEATS_ITER3_CHECKPOINT  = REPO_ROOT / "BEATs_iter3.pt"
_BEATS_ITER3P_CHECKPOINT = REPO_ROOT / "BEATs_iter3_plus_AS2M.pt"
_FISHER_SMALL_CHECKPOINT = REPO_ROOT / "FISHER-small.pt"

MODEL_REGISTRY = {
    "dasheng_base":  lambda: DashengExtractor(variant="base"),
    "dasheng_06B":   lambda: DashengExtractor(variant="06B"),
    "dasheng_12B":   lambda: DashengExtractor(variant="12B"),
    "audiomae":      lambda: AudioMAEExtractor(path=str(_AUDIOMAE_CHECKPOINT)),
    "beats_iter3":   lambda: BEATsExtractor(path=str(_BEATS_ITER3_CHECKPOINT),  name="beats_iter3"),
    "beats_iter3+":  lambda: BEATsExtractor(path=str(_BEATS_ITER3P_CHECKPOINT), name="beats_iter3+"),
    "fisher_small":  lambda: FISHERExtractor(path=str(_FISHER_SMALL_CHECKPOINT), name="fisher_small_4band", freq_bins=200),
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

_DEFAULT_N_SHOTS = [1, 2, 5, 10, 20, 50, 100, 200]

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
        metavar="DATASET",
    )
    p.add_argument(
        "--probes", nargs="+", choices=PROBE_REGISTRY,
        default=["knn", "linear"],
    )
    p.add_argument(
        "--n-shots", nargs="+", type=int, default=_DEFAULT_N_SHOTS,
        metavar="N", help="Shot counts to evaluate (default: 1 2 5 10 20 50 100 200)",
    )
    p.add_argument(
        "--n-repeats", type=int, default=10,
        help="Repetitions per fold per shot level (default: 10)",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "few_shot"),
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
    logger.info(f"Shot levels: {args.n_shots}  |  Repeats: {args.n_repeats}")

    for model_name in args.model:
        for dataset_name in args.dataset:
            model   = MODEL_REGISTRY[model_name]()
            dataset = DATASET_REGISTRY[dataset_name]()
            probes  = [PROBE_REGISTRY[p]() for p in args.probes]

            result = run_few_shot_evaluation(
                model=model,
                dataset=dataset,
                probes=probes,
                device=device,
                batch_size=args.batch_size,
                use_cache=not args.no_cache,
                n_shots_list=args.n_shots,
                n_repeats=args.n_repeats,
            )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = output_dir / f"{model_name}__{dataset_name}__{ts}.json"
            out.write_text(json.dumps(result, indent=2))
            logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
