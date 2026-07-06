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
from eval.datasets.glasgow import (
    GLASGOW_MATURE_YOUNG_EXCLUDE,
    GlasgowDataset,
    HAR5_EXCLUDE,
)
from eval.datasets.mad import GLASGOW_OVERLAP_ACTIVITIES, MADDataset
from eval.few_shot import run_few_shot_evaluation
from eval.models.audiomae import AudioMAEExtractor
from eval.models.beats import BEATsExtractor
from eval.models.dasheng import DashengExtractor
from eval.models.dinov3 import DINOv3Extractor
from eval.models.fisher import FISHERExtractor
from eval.models.mae_imagenet import MAEImageNetExtractor
from eval.models.resnet import build_resnet, preprocess_batch
from eval.models.vit_imagenet import ViTImageNetExtractor
from eval.models.vit_imagenet_lora import build_vit_imagenet_lora
from eval.models.whisper import WhisperExtractor
from eval.models.qwen2_audio import Qwen2AudioExtractor
from eval.models.vjepa2 import VJEPA21Extractor
from eval.probes.knn import KNNProbe
from eval.probes.linear import LinearProbe
from eval.supervised import run_supervised_few_shot_evaluation
from loguru import logger

# ── Registries (keep in sync with run_eval.py) ────────────────────────────────

_AUDIOMAE_CHECKPOINT     = REPO_ROOT / "pretrained.pth"
_BEATS_ITER3_CHECKPOINT  = REPO_ROOT / "BEATs_iter3.pt"
_BEATS_ITER3P_CHECKPOINT = REPO_ROOT / "BEATs_iter3_plus_AS2M.pt"
_FISHER_SMALL_CHECKPOINT = REPO_ROOT / "FISHER-small.pt"
_VJEPA21_CHECKPOINT      = REPO_ROOT / "vjepa2_1_vitb_dist_vitG_384.pt"

MODEL_REGISTRY = {
    "dasheng_base":  lambda: DashengExtractor(variant="base"),
    "dasheng_06B":   lambda: DashengExtractor(variant="06B"),
    "dasheng_12B":   lambda: DashengExtractor(variant="12B"),
    "audiomae":      lambda: AudioMAEExtractor(path=str(_AUDIOMAE_CHECKPOINT)),
    "beats_iter3":   lambda: BEATsExtractor(path=str(_BEATS_ITER3_CHECKPOINT),  name="beats_iter3"),
    "beats_iter3+":  lambda: BEATsExtractor(path=str(_BEATS_ITER3P_CHECKPOINT), name="beats_iter3+"),
    "fisher_small":  lambda: FISHERExtractor(path=str(_FISHER_SMALL_CHECKPOINT), name="fisher_small_4band", freq_bins=200),
    "mae_imagenet":      lambda: MAEImageNetExtractor(),
    "whisper_small":     lambda: WhisperExtractor(variant="small"),
    "whisper_large_v3":  lambda: WhisperExtractor(variant="large_v3"),
    "qwen2_audio":       lambda: Qwen2AudioExtractor(),
    "dinov3_vits16":     lambda: DINOv3Extractor(variant="vits16"),
    "dinov3_vitb16":     lambda: DINOv3Extractor(variant="vitb16"),
    "vit_small_imagenet": lambda: ViTImageNetExtractor(variant="vits16"),
    "vit_base_imagenet":  lambda: ViTImageNetExtractor(variant="vitb16"),
    "vjepa2.1_vitb":     lambda: VJEPA21Extractor(path=str(_VJEPA21_CHECKPOINT)),
}

SUPERVISED_REGISTRY = {
    "resnet50":                (lambda n: build_resnet("resnet50", n),                                  preprocess_batch),
    "vit_base_imagenet_lora":  (lambda n: build_vit_imagenet_lora(n, variant="vitb16"),                 preprocess_batch),
    "vit_base_imagenet_dora":  (lambda n: build_vit_imagenet_lora(n, variant="vitb16", use_dora=True),  preprocess_batch),
    "vit_small_imagenet_lora": (lambda n: build_vit_imagenet_lora(n, variant="vits16"),                 preprocess_batch),
    "vit_small_imagenet_dora": (lambda n: build_vit_imagenet_lora(n, variant="vits16", use_dora=True),  preprocess_batch),
}

_MAD_ROOT = REPO_ROOT / "data" / "MAD"

DATASET_REGISTRY = {
    "glasgow":   lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow"),
    "glasgow_5":  lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow", exclude_classes=HAR5_EXCLUDE),
    "glasgow_young":  lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow", datasets=[1, 2, 3, 4, 5], subset_name="young"),
    # glasgow_mature = D6-7 minus 5 young-aged participants (see
    # GLASGOW_MATURE_YOUNG_EXCLUDE).  Two of those pids are also present
    # in glasgow_young (genuine duplicates), the other three are unique
    # to D6-7 but young; both groups inflate cohort-shift scores when
    # left in.  Drop them explicitly here so the dataset name keeps
    # meaning "older cohort, no leakage".
    "glasgow_mature": lambda: GlasgowDataset(
        REPO_ROOT / "data" / "Glasgow",
        datasets=[6, 7],
        exclude_dpids=GLASGOW_MATURE_YOUNG_EXCLUDE,
        subset_name="mature",
    ),
    "esc50":     lambda: ESC50Dataset(REPO_ROOT / "data" / "ESC-50-master"),
    "mad":       lambda: MADDataset(_MAD_ROOT),
    "mad_5":       lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES),
    "mad_5_sub1":  lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES, subcategories=[1]),
    "mad_5_sub2":  lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES, subcategories=[2]),
    "mad_5_sub3":  lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES, subcategories=[3]),
    "mad_5_sub12": lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES, subcategories=[1, 2]),
    "mad_5_sub23": lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES, subcategories=[2, 3]),
    "mad_sub1":    lambda: MADDataset(_MAD_ROOT, subcategories=[1]),
    "mad_sub2":    lambda: MADDataset(_MAD_ROOT, subcategories=[2]),
    "mad_sub3":    lambda: MADDataset(_MAD_ROOT, subcategories=[3]),
    "mad_sub12":   lambda: MADDataset(_MAD_ROOT, subcategories=[1, 2]),
    "mad_sub23":   lambda: MADDataset(_MAD_ROOT, subcategories=[2, 3]),
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
    all_models = list(MODEL_REGISTRY) + list(SUPERVISED_REGISTRY)
    p.add_argument(
        "--model", nargs="+", choices=all_models, required=True,
        metavar="MODEL", help=f"One or more of: {all_models}",
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
            dataset = DATASET_REGISTRY[dataset_name]()

            if model_name in SUPERVISED_REGISTRY:
                factory, preprocess_fn = SUPERVISED_REGISTRY[model_name]
                result = run_supervised_few_shot_evaluation(
                    model_name=model_name,
                    model_factory=factory,
                    preprocess_fn=preprocess_fn,
                    dataset=dataset,
                    device=device,
                    batch_size=args.batch_size,
                    n_shots_list=args.n_shots,
                    n_repeats=args.n_repeats,
                )
            else:
                model  = MODEL_REGISTRY[model_name]()
                probes = [PROBE_REGISTRY[p]() for p in args.probes]
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
