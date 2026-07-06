#!/usr/bin/env python3
"""One-off analysis: per-subgroup performance on glasgow_mature for
checkpoints fine-tuned on glasgow_young.

Subgroups within glasgow_mature:
  - dup:          (D, pid) where the same physical person also appears in
                  glasgow_young  (pid 8 D06, pid 31 D06).
  - young_unique: young-aged (<40) participants whose pid does not appear
                  in glasgow_young  (pid 22 D06, pid 25 D06, pid 56 D07).
  - old:          everyone else (the actual mature cohort).
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.datasets.glasgow import GlasgowDataset
from eval.models.dinov3_lora import build_dinov3_lora
from eval.models.resnet import build_resnet, preprocess_batch
from eval.supervised import _FoldDataset, _evaluate

CKPT_DIR = REPO_ROOT / "results" / "checkpoints"
SCHEME = "glasgow6"   # checkpoints were trained under this scheme
EPOCHS = 30
N_CLASSES = 6
CLASS_NAMES = [
    "Walking", "Sitting down", "Standing up",
    "Picking up object", "Drinking", "Falling",
]

# Same physical person on both sides (D1-5 ↔ D6-7).
DUP_KEYS = {(6, 8), (6, 31)}
# Young-aged (<40) in D6-7 with no D1-5 counterpart.
YOUNG_UNIQUE_KEYS = {(6, 22), (6, 25), (7, 56)}

_RE_DPID = re.compile(r"^D0(\d)_\dP(\d+)")


def subgroup_of(stem: str) -> str:
    m = _RE_DPID.match(stem)
    if m is None:
        raise ValueError(f"Cannot parse D/pid from {stem}")
    key = (int(m.group(1)), int(m.group(2)))
    if key in DUP_KEYS:
        return "dup"
    if key in YOUNG_UNIQUE_KEYS:
        return "young_unique"
    return "old"


def report(y_true: np.ndarray, y_pred: np.ndarray, tags: np.ndarray) -> None:
    overall_acc = float((y_pred == y_true).mean())
    overall_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    print(
        f"  overall      n={len(y_true):>4}  acc={overall_acc * 100:6.2f}%  "
        f"macro-F1={overall_f1 * 100:6.2f}%"
    )
    for sg in ("dup", "young_unique", "old"):
        mask = tags == sg
        n = int(mask.sum())
        if n == 0:
            print(f"  {sg:<12} n=   0  (empty)")
            continue
        sg_acc = float((y_pred[mask] == y_true[mask]).mean())
        # Restrict macro-F1 to classes actually present (otherwise zero
        # division dominates).  Use labels=present_classes.
        present = sorted(set(y_true[mask].tolist()))
        sg_f1 = float(f1_score(
            y_true[mask], y_pred[mask],
            labels=present, average="macro", zero_division=0,
        ))
        # Per-class breakdown (count and per-class accuracy).
        cls_str = ", ".join(
            f"{CLASS_NAMES[c]}: {(y_pred[mask][y_true[mask] == c] == c).sum()}/"
            f"{(y_true[mask] == c).sum()}"
            for c in present
        )
        print(
            f"  {sg:<12} n={n:>4}  acc={sg_acc * 100:6.2f}%  "
            f"macro-F1={sg_f1 * 100:6.2f}%  | {cls_str}"
        )


def run_model(
    model_name: str,
    factory,
    ckpt_path: Path,
    dataset: GlasgowDataset,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    model = factory(N_CLASSES).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    items, labels = dataset.items()
    indices = np.arange(len(items))
    ds = _FoldDataset(dataset, indices, items, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return _evaluate(model, loader, preprocess_batch, device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = GlasgowDataset(
        REPO_ROOT / "data" / "Glasgow",
        datasets=[6, 7],
        subset_name="mature",
    )
    items, labels = dataset.items()
    tags = np.array([subgroup_of(p.stem) for p in items])
    print(
        f"\nglasgow_mature: n={len(items)}  "
        f"dup={int((tags == 'dup').sum())}  "
        f"young_unique={int((tags == 'young_unique').sum())}  "
        f"old={int((tags == 'old').sum())}\n"
    )

    runs = [
        (
            "resnet50",
            lambda n: build_resnet("resnet50", n),
            CKPT_DIR / f"resnet50_ft_glasgow_young_{SCHEME}_ep{EPOCHS}.pt",
        ),
        (
            "dinov3_lora",
            lambda n: build_dinov3_lora(n),
            CKPT_DIR / f"dinov3_lora_ft_glasgow_young_{SCHEME}_ep{EPOCHS}.pt",
        ),
    ]

    for model_name, factory, ckpt in runs:
        print(f"=== {model_name}  (ckpt: {ckpt.name}) ===")
        if not ckpt.exists():
            print(f"  MISSING: {ckpt}")
            continue
        y_true, y_pred = run_model(model_name, factory, ckpt, dataset, device)
        report(y_true, y_pred, tags)
        print()


if __name__ == "__main__":
    main()
