#!/usr/bin/env python3
"""Sweep LoRA rank on Glasgow (in-domain CV) to select the best rank.

Runs 5-fold cross-validation for each rank, reports accuracy and F1.
The selected rank can then be used for cross-dataset evaluation
without leaking target information.

Example
-------
python scripts/sweep_lora_rank.py
python scripts/sweep_lora_rank.py --ranks 4 8 16 32 --epochs 30
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.datasets.glasgow import GlasgowDataset
from eval.models.dinov3_lora import build_dinov3_lora
from eval.models.resnet import preprocess_batch
from eval.supervised import _FoldDataset, _evaluate, _train_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ranks", nargs="+", type=int, default=[4, 8, 16, 32],
    )
    p.add_argument(
        "--variant", default="vitb16", choices=["vitb16", "vits16"],
        help="DINOv3 backbone variant.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "lora_rank_sweep"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"sweep_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_path, level="DEBUG")

    dataset = GlasgowDataset(REPO_ROOT / "data" / "Glasgow")
    items, labels = dataset.items()
    folds = list(dataset.cv_splits())
    criterion = nn.CrossEntropyLoss()

    all_results = {}

    for rank in args.ranks:
        logger.info(f"\n{'='*60}")
        logger.info(f"LoRA rank = {rank}")
        logger.info(f"{'='*60}")

        fold_accs, fold_f1s = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            model = build_dinov3_lora(
                dataset.n_classes, variant=args.variant,
                rank=rank, alpha=rank * 2,
            ).to(device)

            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=args.lr, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs,
            )

            train_ds = _FoldDataset(dataset, train_idx, items, labels)
            test_ds = _FoldDataset(dataset, test_idx, items, labels)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                shuffle=True, num_workers=4,
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size,
                shuffle=False, num_workers=4,
            )

            for epoch in tqdm(
                range(args.epochs),
                desc=f"r={rank} fold {fold_idx + 1}",
                leave=False,
            ):
                loss = _train_one_epoch(
                    model, train_loader, optimizer, criterion,
                    preprocess_batch, device,
                )
                scheduler.step()
                if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                    logger.debug(
                        f"  [r={rank}] fold {fold_idx + 1} "
                        f"epoch {epoch + 1}/{args.epochs}  loss={loss:.4f}"
                    )

            y_true, y_pred = _evaluate(
                model, test_loader, preprocess_batch, device,
            )
            acc = float((y_pred == y_true).mean())
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            fold_accs.append(acc)
            fold_f1s.append(f1)
            logger.info(
                f"  [r={rank}] fold {fold_idx + 1}: "
                f"acc={acc * 100:.2f}%  F1={f1 * 100:.2f}%"
            )

            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))

        all_results[rank] = {
            "rank": rank,
            "fold_accs": fold_accs,
            "fold_f1s": fold_f1s,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        }

        logger.info(
            f"[r={rank}] Glasgow CV: "
            f"acc={mean_acc * 100:.2f}% +/- {std_acc * 100:.2f}%  "
            f"F1={mean_f1 * 100:.2f}% +/- {std_f1 * 100:.2f}%"
        )

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LoRA Rank Sweep — Glasgow 5-fold CV")
    print("=" * 60)
    print(f"{'Rank':>6}  {'Acc (%)':>12}  {'F1 (%)':>12}")
    print("-" * 36)

    best_rank, best_f1 = None, -1.0
    for rank in args.ranks:
        r = all_results[rank]
        print(
            f"{rank:>6}  "
            f"{r['mean_acc'] * 100:5.2f} +/- {r['std_acc'] * 100:4.2f}  "
            f"{r['mean_f1'] * 100:5.2f} +/- {r['std_f1'] * 100:4.2f}"
        )
        if r["mean_f1"] > best_f1:
            best_f1 = r["mean_f1"]
            best_rank = rank

    print("-" * 36)
    print(f"Best rank by F1: {best_rank}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"lora_rank_sweep_{args.variant}_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
