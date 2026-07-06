#!/usr/bin/env python3
"""Cross-dataset evaluation: train on one dataset, test on another.

Examples
--------
# Train on Glasgow, test on MAD-5
python scripts/run_cross_eval.py \\
    --model dasheng_base \\
    --train-dataset glasgow \\
    --test-dataset mad_5

# Multiple models and test datasets
python scripts/run_cross_eval.py \\
    --model dasheng_base dasheng_06B \\
    --train-dataset glasgow \\
    --test-dataset mad_5 mad_5_sub1

# Both directions
python scripts/run_cross_eval.py \\
    --model dasheng_base \\
    --train-dataset glasgow \\
    --test-dataset mad_5 \\
    --bidirectional
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.cross_evaluation import (
    CROSS_LABEL_SCHEMES,
    run_cross_evaluation,
    run_supervised_cross_evaluation,
)
from eval.datasets.esc50 import ESC50Dataset
from eval.datasets.glasgow import (
    GLASGOW_MATURE_YOUNG_EXCLUDE,
    GlasgowDataset,
    HAR5_EXCLUDE,
)
from eval.datasets.mad import GLASGOW_OVERLAP_ACTIVITIES, MADDataset
from eval.models.dasheng_lora import (
    build_dasheng_lora,
    preprocess_batch as dasheng_preprocess,
)
from eval.models.dinov3_lora import build_dinov3_lora
from eval.models.resnet import build_resnet, preprocess_batch
from eval.models.vit_imagenet import ViTImageNetExtractor
from eval.models.vit_imagenet_lora import build_vit_imagenet_lora
from eval.models.audiomae import AudioMAEExtractor
from eval.models.beats import BEATsExtractor
from eval.models.dasheng import DashengExtractor
from eval.models.dinov3 import DINOv3Extractor
from eval.models.fisher import FISHERExtractor
from eval.models.mae_imagenet import MAEImageNetExtractor
from eval.models.qwen2_audio import Qwen2AudioExtractor
from eval.models.vjepa2 import VJEPA21Extractor
from eval.models.whisper import WhisperExtractor
from eval.probes.knn import KNNProbe
from eval.probes.linear import LinearProbe
from loguru import logger

# ── Registries (keep in sync with run_eval.py) ──────────────────────────────

_AUDIOMAE_CHECKPOINT      = REPO_ROOT / "pretrained.pth"
_BEATS_ITER3_CHECKPOINT   = REPO_ROOT / "BEATs_iter3.pt"
_BEATS_ITER3P_CHECKPOINT  = REPO_ROOT / "BEATs_iter3_plus_AS2M.pt"
_FISHER_SMALL_CHECKPOINT  = REPO_ROOT / "FISHER-small.pt"
_VJEPA21_CHECKPOINT       = REPO_ROOT / "vjepa2_1_vitb_dist_vitG_384.pt"
_MAD_ROOT                 = REPO_ROOT / "data" / "MAD"

MODEL_REGISTRY = {
    "dasheng_base":     lambda: DashengExtractor(variant="base"),
    "dasheng_06B":      lambda: DashengExtractor(variant="06B"),
    "dasheng_12B":      lambda: DashengExtractor(variant="12B"),
    "audiomae":         lambda: AudioMAEExtractor(path=str(_AUDIOMAE_CHECKPOINT)),
    "beats_iter3":      lambda: BEATsExtractor(path=str(_BEATS_ITER3_CHECKPOINT),  name="beats_iter3"),
    "beats_iter3+":     lambda: BEATsExtractor(path=str(_BEATS_ITER3P_CHECKPOINT), name="beats_iter3+"),
    "fisher_small":     lambda: FISHERExtractor(path=str(_FISHER_SMALL_CHECKPOINT), name="fisher_small_4band", freq_bins=200),
    "mae_imagenet":     lambda: MAEImageNetExtractor(),
    "whisper_small":    lambda: WhisperExtractor(variant="small"),
    "whisper_large_v3": lambda: WhisperExtractor(variant="large_v3"),
    "qwen2_audio":      lambda: Qwen2AudioExtractor(),
    "dinov3_vits16":    lambda: DINOv3Extractor(variant="vits16"),
    "dinov3_vitb16":    lambda: DINOv3Extractor(variant="vitb16"),
    "vit_small_imagenet": lambda: ViTImageNetExtractor(variant="vits16"),
    "vit_base_imagenet":  lambda: ViTImageNetExtractor(variant="vitb16"),
    "vjepa2.1_vitb":    lambda: VJEPA21Extractor(path=str(_VJEPA21_CHECKPOINT)),
}

DATASET_REGISTRY = {
    "glasgow":    lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow"),
    "glasgow_5":   lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow", exclude_classes=HAR5_EXCLUDE),
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
    "esc50":      lambda: ESC50Dataset(REPO_ROOT / "data" / "ESC-50-master"),
    "mad":        lambda: MADDataset(_MAD_ROOT),
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

SUPERVISED_REGISTRY = {
    "resnet50": (lambda n: build_resnet("resnet50", n), preprocess_batch),
    # LoRA adapters
    "dinov3_lora":             (lambda n: build_dinov3_lora(n),                                         preprocess_batch),
    "dinov3_vits16_lora":      (lambda n: build_dinov3_lora(n, variant="vits16"),                       preprocess_batch),
    "vit_base_imagenet_lora":  (lambda n: build_vit_imagenet_lora(n, variant="vitb16"),                 preprocess_batch),
    "vit_small_imagenet_lora": (lambda n: build_vit_imagenet_lora(n, variant="vits16"),                 preprocess_batch),
    "dasheng_lora":            (lambda n: build_dasheng_lora(n, variant="base"),                        dasheng_preprocess),
    "dasheng_06B_lora":        (lambda n: build_dasheng_lora(n, variant="06B"),                         dasheng_preprocess),
    "dasheng_12B_lora":        (lambda n: build_dasheng_lora(n, variant="12B"),                         dasheng_preprocess),
    # DoRA (weight-decomposed LoRA) adapters — same target modules,
    # use_dora=True. Checkpoints are NOT interchangeable with LoRA.
    "dinov3_dora":             (lambda n: build_dinov3_lora(n, use_dora=True),                          preprocess_batch),
    "dinov3_vits16_dora":      (lambda n: build_dinov3_lora(n, variant="vits16", use_dora=True),        preprocess_batch),
    "vit_base_imagenet_dora":  (lambda n: build_vit_imagenet_lora(n, variant="vitb16", use_dora=True),  preprocess_batch),
    "vit_small_imagenet_dora": (lambda n: build_vit_imagenet_lora(n, variant="vits16", use_dora=True),  preprocess_batch),
    "dasheng_dora":            (lambda n: build_dasheng_lora(n, variant="base", use_dora=True),         dasheng_preprocess),
    "dasheng_06B_dora":        (lambda n: build_dasheng_lora(n, variant="06B",  use_dora=True),         dasheng_preprocess),
    "dasheng_12B_dora":        (lambda n: build_dasheng_lora(n, variant="12B",  use_dora=True),         dasheng_preprocess),
}

PROBE_REGISTRY = {
    "knn":    lambda: KNNProbe(k=10, metric="cosine"),
    "linear": lambda: LinearProbe(C=1.0),
}

# ─�� CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    all_models = list(MODEL_REGISTRY) + list(SUPERVISED_REGISTRY)
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", nargs="+", choices=all_models, required=True,
        metavar="MODEL", help=f"One or more of: {all_models}",
    )
    p.add_argument(
        "--train-dataset", choices=DATASET_REGISTRY, required=True,
        metavar="DATASET",
    )
    p.add_argument(
        "--test-dataset", nargs="+", choices=DATASET_REGISTRY, required=True,
        metavar="DATASET",
    )
    p.add_argument(
        "--probes", nargs="+", choices=PROBE_REGISTRY,
        default=["knn", "linear"],
        metavar="PROBE",
    )
    p.add_argument(
        "--bidirectional", action="store_true",
        help="Also run the reverse direction (test→train).",
    )
    p.add_argument(
        "--center-per-dataset", action="store_true",
        help="Subtract each dataset's feature mean before probing "
             "(removes first-order distribution shift).",
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
        default=str(REPO_ROOT / "results" / "cross_eval"),
    )
    p.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "results" / "checkpoints"),
        help="Where to save/load fine-tuned supervised checkpoints. "
             "Shared with run_cross_few_shot.py so an FT done by either "
             "script is reusable by the other (same naming: "
             "{model}_ft_{source}_{scheme}_ep{epochs}.pt).",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Fine-tuning epochs for supervised models.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for supervised fine-tuning.",
    )
    p.add_argument(
        "--scheme", choices=list(CROSS_LABEL_SCHEMES), default=None,
        help="Force a specific label scheme. By default the first scheme "
             "covering both datasets is used. Required to disambiguate when "
             "more than one scheme matches (e.g. glasgow→mad_5 fits "
             "har5/mad10/glasgow6).",
    )
    return p.parse_args()


def print_cross_results(results: dict) -> None:
    """Pretty-print a cross-evaluation result."""
    print(f"\nModel: {results['model']}")
    print(
        f"Train: {results['train_dataset']} ({results['n_train']} samples)  →  "
        f"Test: {results['test_dataset']} ({results['n_test']} samples)"
    )
    print(f"Label scheme: {results['label_scheme']} ({results['n_classes']} classes)")
    print("-" * 56)
    print(f"  {'probe':<20}  {'accuracy':>14}   {'macro-F1':>14}")
    print("-" * 56)
    for probe_name, probe_res in results["probes"].items():
        acc = f"{probe_res['acc'] * 100:.2f}%"
        f1 = f"{probe_res['f1'] * 100:.2f}%"
        print(f"  {probe_name:<20}  {acc:>14}   {f1:>14}")
    print()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_path, level="DEBUG")
    logger.info(f"Device: {device}  |  log → {log_path}")

    # Build list of (train, test) pairs.
    pairs = [
        (args.train_dataset, test_name)
        for test_name in args.test_dataset
    ]
    if args.bidirectional:
        pairs += [
            (test_name, args.train_dataset)
            for test_name in args.test_dataset
        ]

    for model_name in args.model:
        for train_name, test_name in pairs:
            train_ds = DATASET_REGISTRY[train_name]()
            test_ds = DATASET_REGISTRY[test_name]()

            if model_name in SUPERVISED_REGISTRY:
                factory, preprocess_fn = SUPERVISED_REGISTRY[model_name]
                result = run_supervised_cross_evaluation(
                    model_name=model_name,
                    model_factory=factory,
                    preprocess_fn=preprocess_fn,
                    train_dataset=train_ds,
                    test_dataset=test_ds,
                    device=device,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    epochs=args.epochs,
                    checkpoint_dir=Path(args.checkpoint_dir),
                    scheme=args.scheme,
                )
            else:
                model = MODEL_REGISTRY[model_name]()
                probes = [PROBE_REGISTRY[p]() for p in args.probes]
                result = run_cross_evaluation(
                    model=model,
                    train_dataset=train_ds,
                    test_dataset=test_ds,
                    probes=probes,
                    device=device,
                    batch_size=args.batch_size,
                    use_cache=not args.no_cache,
                    center_per_dataset=args.center_per_dataset,
                    scheme=args.scheme,
                )

            print_cross_results(result)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = output_dir / f"{model_name}__{train_name}_to_{test_name}__{ts}.json"
            out.write_text(json.dumps(result, indent=2))
            logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
