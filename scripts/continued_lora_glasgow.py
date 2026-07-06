#!/usr/bin/env python3
"""Continued LoRA fine-tuning on glasgow_mature few-shot subsets.

Quick standalone experiment to compare against the standard
"strip head + linear probe" cross few-shot pipeline.

Pipeline per fold:
  1. Build DINOv3+LoRA with 6-class head (glasgow6 scheme).
  2. Load FT checkpoint trained on glasgow_young.
  3. Sample N shots/class from the fold's train split (stratified by
     group, like the standard pipeline — same fold splits are reused).
  4. Continue training BOTH LoRA adapters and head on those N shots.
  5. Argmax prediction on the held-out fold (no linear probe).

Repeats per shot level use different sampling seeds, identical to
`run_cross_few_shot_evaluation` so numbers are directly comparable.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from eval.datasets.glasgow import GLASGOW_MATURE_YOUNG_EXCLUDE, GlasgowDataset
from eval.few_shot import _stratified_sample
from eval.models.dinov3_lora import build_dinov3_lora
from eval.models.resnet import preprocess_batch  # same preproc used in cross pipeline
from eval.supervised import _evaluate, _FoldDataset, _train_one_epoch

# ── Config ────────────────────────────────────────────────────────────
N_CLASSES = 6                       # glasgow6 scheme
SCHEME = "glasgow6"
SOURCE = "glasgow_young"
TARGET = "glasgow_mature"
CKPT = REPO_ROOT / "results" / "checkpoints" / f"dinov3_lora_ft_{SOURCE}_{SCHEME}_ep30.pt"

N_SHOTS_LIST = [1, 2, 5, 10, 20]
N_REPEATS = 10
N_FOLDS = 5
SEED = 42

# Continued-FT hyperparameters: lower LR than initial FT (1e-4), short
# schedule.  These are deliberately conservative — the goal is "nudge
# the source-trained model toward the target", not retrain from scratch.
LR = 5e-5
EPOCHS = 20
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8

OUT_DIR = REPO_ROOT / "results" / "cross_few_shot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LINEAR_REF = OUT_DIR / "dinov3_lora_ft_glasgow_young__glasgow_mature__20260429_173829.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if not CKPT.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {CKPT}")

    target_ds = GlasgowDataset(
        REPO_ROOT / "data" / "Glasgow",
        datasets=[6, 7],
        exclude_dpids=GLASGOW_MATURE_YOUNG_EXCLUDE,
        subset_name="mature",
    )
    items, labels = target_ds.items()
    groups = target_ds.groups
    classes = np.unique(labels)
    logger.info(
        f"Target {target_ds.name}: {len(labels)} samples, "
        f"classes={classes.tolist()}"
    )

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(sgkf.split(np.zeros(len(labels)), labels, groups))

    # Cache initial state once — reload between (fold, n_shots, repeat) trials
    # so each trial starts from the source-FT'd weights.
    base_state = torch.load(CKPT, map_location=device, weights_only=True)
    criterion = nn.CrossEntropyLoss()

    # raw_scores[n_shots] = list of (acc, f1)
    raw_scores: dict[int, list[tuple[float, float]]] = {n: [] for n in N_SHOTS_LIST}

    t0 = time.time()
    n_trials_total = len(folds) * len(N_SHOTS_LIST) * N_REPEATS
    trial_idx = 0

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        test_ds = _FoldDataset(target_ds, test_idx, items, labels)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        )

        for n_shots in N_SHOTS_LIST:
            for rep in range(N_REPEATS):
                trial_idx += 1
                rng = np.random.default_rng(
                    SEED * 10_000 + fold_idx * 1_000 + rep,
                )
                sub_idx = train_idx[
                    _stratified_sample(labels[train_idx], classes, n_shots, rng)
                ]

                train_ds = _FoldDataset(target_ds, sub_idx, items, labels)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=min(BATCH_SIZE, len(train_ds)),
                    shuffle=True,
                    num_workers=2,
                )

                # Fresh model per trial, reloaded from source checkpoint.
                model = build_dinov3_lora(N_CLASSES).to(device)
                model.load_state_dict(base_state)

                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=LR, weight_decay=WEIGHT_DECAY,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS,
                )
                for _ in range(EPOCHS):
                    _train_one_epoch(
                        model, train_loader, optimizer, criterion,
                        preprocess_batch, device,
                    )
                    scheduler.step()

                y_true, y_pred = _evaluate(
                    model, test_loader, preprocess_batch, device,
                )
                acc = float((y_pred == y_true).mean())
                present = sorted(set(y_true.tolist()))
                f1 = float(f1_score(
                    y_true, y_pred,
                    labels=present, average="macro", zero_division=0,
                ))
                raw_scores[n_shots].append((acc, f1))

                elapsed = time.time() - t0
                eta = elapsed / trial_idx * (n_trials_total - trial_idx)
                logger.info(
                    f"  [fold {fold_idx + 1}/{N_FOLDS} | "
                    f"n_shots={n_shots} | rep={rep + 1}/{N_REPEATS}] "
                    f"acc={acc * 100:.2f}%  F1={f1 * 100:.2f}%  "
                    f"(elapsed {elapsed / 60:.1f}m, eta {eta / 60:.1f}m)"
                )

                del model, optimizer, scheduler
                torch.cuda.empty_cache()

    # Aggregate
    probe_data = []
    for n_shots in N_SHOTS_LIST:
        accs = [s[0] for s in raw_scores[n_shots]]
        f1s = [s[1] for s in raw_scores[n_shots]]
        probe_data.append({
            "n_shots": n_shots,
            "n_total_train": n_shots * int(len(classes)),
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
        })

    result = {
        "model": f"dinov3_lora_continued_{SOURCE}",
        "source_dataset": SOURCE,
        "dataset": TARGET,
        "label_scheme": SCHEME,
        "n_classes": int(len(classes)),
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "config": {
            "lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY, "ckpt": str(CKPT.name),
        },
        "probes": {"continued_lora": probe_data},
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"dinov3_lora_continued_{SOURCE}__{TARGET}__{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info(f"Saved → {out}")

    # ── Compare with logged linear-probe results ───────────────────────
    print()
    print(f"{'n_shots':<10} {'continued_lora':<25} {'linear (logged)':<25} {'knn (logged)':<25}")
    print("-" * 90)
    ref = json.loads(LINEAR_REF.read_text()) if LINEAR_REF.exists() else None
    lin = {e["n_shots"]: e for e in ref["probes"].get("linear", [])} if ref else {}
    knn = {e["n_shots"]: e for e in ref["probes"].get("knn_k10", [])} if ref else {}
    for entry in probe_data:
        n = entry["n_shots"]
        cur = f"{entry['mean_acc'] * 100:5.2f}±{entry['std_acc'] * 100:4.2f}%"
        l = lin.get(n)
        k = knn.get(n)
        l_s = f"{l['mean_acc'] * 100:5.2f}±{l['std_acc'] * 100:4.2f}%" if l else "—"
        k_s = f"{k['mean_acc'] * 100:5.2f}±{k['std_acc'] * 100:4.2f}%" if k else "—"
        print(f"{n:<10} {cur:<25} {l_s:<25} {k_s:<25}")


if __name__ == "__main__":
    main()
