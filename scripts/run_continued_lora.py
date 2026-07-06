#!/usr/bin/env python3
"""Continued-LoRA cross-dataset few-shot evaluation.

Load a source-fine-tuned LoRA checkpoint (LoRA adapters + classification
head) and continue training on N-shot subsets of the target dataset,
instead of stripping the head and fitting a frozen-feature linear probe.

The source checkpoint must already exist in ``--checkpoint-dir`` under
the standard naming used by ``run_cross_few_shot.py``:

    {model}_ft_{source}_{scheme}_ep{source_epochs}.pt

Output JSON is written to ``--output-dir`` with the same shape as
``run_cross_few_shot_evaluation`` (``probes`` keyed by ``continued_lora``).

Examples
--------
    python scripts/run_continued_lora.py \\
        --model dinov3_lora \\
        --source-dataset glasgow_young \\
        --target-dataset glasgow_mature
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
load_dotenv(REPO_ROOT / ".env")

from eval.cross_evaluation import (
    CROSS_LABEL_SCHEMES, _remap_dataset_labels, find_label_scheme,
)
from eval.few_shot import _stratified_sample
from eval.supervised import _FoldDataset, _evaluate, _train_one_epoch
from run_cross_few_shot import DATASET_REGISTRY, SUPERVISED_REGISTRY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=SUPERVISED_REGISTRY)
    p.add_argument("--source-dataset", required=True, choices=DATASET_REGISTRY)
    p.add_argument("--target-dataset", required=True, choices=DATASET_REGISTRY)
    p.add_argument(
        "--scheme", choices=list(CROSS_LABEL_SCHEMES), default=None,
        help="Force a label scheme. Default: auto-detect from registry.",
    )
    p.add_argument("--n-shots", nargs="+", type=int, default=[1, 2, 5, 10])
    p.add_argument(
        "--n-repeats", nargs="+", type=int, default=[30, 20, 10, 10],
        help="Repeats per shot level. Either one value (used for all "
             "shots) or one per --n-shots entry. Default scales inversely "
             "with shot count to equalize SEM across the curve.",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--source-epochs", type=int, default=30,
        help="Epochs used during source FT (for checkpoint name lookup).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "results" / "checkpoints"),
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "cross_few_shot"),
    )
    return p.parse_args()


def _resolve_repeats(n_shots: list[int], n_repeats: list[int]) -> dict[int, int]:
    """Map each shot level to its repeat count (broadcast if length 1)."""
    if len(n_repeats) == 1:
        return {s: n_repeats[0] for s in n_shots}
    if len(n_repeats) != len(n_shots):
        raise ValueError(
            f"--n-repeats must have length 1 or {len(n_shots)} "
            f"(got {len(n_repeats)})"
        )
    return dict(zip(n_shots, n_repeats))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts0 = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"continued_lora_{ts0}.log"
    logger.add(log_path, level="DEBUG")
    logger.info(f"Device: {device}  |  log → {log_path}")

    # ── Build datasets and resolve scheme ────────────────────────────────
    source_ds = DATASET_REGISTRY[args.source_dataset]()
    target_ds = DATASET_REGISTRY[args.target_dataset]()
    scheme_name, scheme_def = find_label_scheme(
        source_ds.name, target_ds.name, args.scheme,
    )
    remaps = scheme_def["dataset_remaps"]
    n_classes_scheme = len(scheme_def["class_names"])
    logger.info(
        f"Continued LoRA: {source_ds.name} → {target_ds.name}  "
        f"scheme={scheme_name} ({n_classes_scheme} classes)"
    )

    # ── Locate and load source checkpoint ────────────────────────────────
    ckpt_path = (
        Path(args.checkpoint_dir)
        / f"{args.model}_ft_{source_ds.name}_{scheme_name}_ep{args.source_epochs}.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found: {ckpt_path}\n"
            f"Run scripts/run_cross_few_shot.py first to fine-tune."
        )
    logger.info(f"Source ckpt: {ckpt_path.name}")
    factory, preprocess_fn = SUPERVISED_REGISTRY[args.model]
    base_state = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Build the model ONCE outside the trial loop — re-instantiating the
    # PEFT-wrapped HF backbone hundreds of times causes CUDA fragmentation
    # that accumulates faster than `empty_cache()` can release.
    model = factory(n_classes_scheme).to(device)

    # ── Prep target items + remap labels into scheme space ───────────────
    items_all, labels_raw = target_ds.items()
    indices_keep, labels_full = _remap_dataset_labels(
        labels_raw, remaps[target_ds.name],
    )
    items = [items_all[i] for i in indices_keep]
    labels = labels_full[indices_keep]
    groups = target_ds.groups
    if groups is not None:
        groups = groups[indices_keep]
    classes = np.unique(labels)
    logger.info(
        f"Target after remap: {len(labels)} samples, "
        f"classes_present={classes.tolist()}"
    )

    # ── Folds ────────────────────────────────────────────────────────────
    if groups is not None:
        sgkf = StratifiedGroupKFold(
            n_splits=args.n_folds, shuffle=True, random_state=42,
        )
        folds = list(sgkf.split(np.zeros(len(labels)), labels, groups))
    else:
        skf = StratifiedKFold(
            n_splits=args.n_folds, shuffle=True, random_state=42,
        )
        folds = list(skf.split(np.zeros(len(labels)), labels))

    # ── Continued-FT loop ────────────────────────────────────────────────
    repeats_by_shot = _resolve_repeats(args.n_shots, args.n_repeats)
    logger.info(
        "Repeats per shot: "
        + ", ".join(f"{s}→{repeats_by_shot[s]}" for s in args.n_shots)
    )
    criterion = nn.CrossEntropyLoss()
    raw_scores: dict[int, list[tuple[float, float]]] = {
        n: [] for n in args.n_shots
    }
    n_trials_total = len(folds) * sum(repeats_by_shot.values())
    trial_idx = 0
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        test_loader = DataLoader(
            _FoldDataset(target_ds, test_idx, items, labels),
            batch_size=args.batch_size, shuffle=False, num_workers=4,
        )

        for n_shots in args.n_shots:
            n_rep = repeats_by_shot[n_shots]
            for rep in range(n_rep):
                trial_idx += 1
                rng = np.random.default_rng(
                    args.seed * 10_000 + fold_idx * 1_000 + rep,
                )
                sub_idx = train_idx[
                    _stratified_sample(labels[train_idx], classes, n_shots, rng)
                ]
                train_loader = DataLoader(
                    _FoldDataset(target_ds, sub_idx, items, labels),
                    batch_size=min(args.batch_size, len(sub_idx)),
                    shuffle=True, num_workers=2,
                )

                model.load_state_dict(base_state)

                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs,
                )
                for _ in range(args.epochs):
                    _train_one_epoch(
                        model, train_loader, optimizer, criterion,
                        preprocess_fn, device,
                    )
                    scheduler.step()

                y_true, y_pred = _evaluate(
                    model, test_loader, preprocess_fn, device,
                )
                acc = float((y_pred == y_true).mean())
                present = sorted(set(y_true.tolist()))
                f1 = float(f1_score(
                    y_true, y_pred, labels=present,
                    average="macro", zero_division=0,
                ))
                raw_scores[n_shots].append((acc, f1))

                elapsed = time.time() - t0
                eta = elapsed / trial_idx * (n_trials_total - trial_idx)
                logger.info(
                    f"  [fold {fold_idx + 1}/{len(folds)} | "
                    f"n_shots={n_shots} | rep={rep + 1}/{n_rep}] "
                    f"acc={acc * 100:.2f}%  F1={f1 * 100:.2f}%  "
                    f"(elapsed {elapsed / 60:.1f}m, eta {eta / 60:.1f}m)"
                )

                del optimizer, scheduler
                torch.cuda.empty_cache()

    # ── Aggregate + save ─────────────────────────────────────────────────
    display_name = f"{args.model}_continued_{source_ds.name}"
    probe_data = []
    for n_shots in args.n_shots:
        accs = [s[0] for s in raw_scores[n_shots]]
        f1s = [s[1] for s in raw_scores[n_shots]]
        entry = {
            "n_shots": n_shots,
            "n_total_train": n_shots * int(len(classes)),
            "n_repeats": repeats_by_shot[n_shots],
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
        }
        probe_data.append(entry)
        logger.info(
            f"[{display_name}] [{target_ds.name}] continued_lora "
            f"{n_shots}-shot: "
            f"acc={entry['mean_acc'] * 100:.2f}% ± {entry['std_acc'] * 100:.2f}%  "
            f"F1={entry['mean_f1'] * 100:.2f}% ± {entry['std_f1'] * 100:.2f}%"
        )

    result = {
        "model": display_name,
        "source_dataset": source_ds.name,
        "dataset": target_ds.name,
        "label_scheme": scheme_name,
        "n_classes": int(len(classes)),
        "n_folds": len(folds),
        "n_repeats": repeats_by_shot,
        "config": {
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "ckpt": ckpt_path.name,
        },
        "probes": {"continued_lora": probe_data},
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"{display_name}__{target_ds.name}__{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
