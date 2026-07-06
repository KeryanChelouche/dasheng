#!/usr/bin/env python3
"""Quick test: Dasheng-base on Glasgow with different magnitude transforms.

Tests raw (no transform), log1p, AmplitudeToDB, and AmplitudeToDB+BN-adapt,
all with StandardScaler + cosine KNN k=10.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torchF
import torchaudio.transforms as T
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch.amp import autocast
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dasheng import dasheng_base_spectrogram

# ── Load Glasgow ─────────────────────────────────────────────────────────────
DATA_ROOT = REPO_ROOT / "data" / "Glasgow"
files, labels = [], []
for class_id in range(1, 7):
    for f in sorted((DATA_ROOT / str(class_id)).glob("*.npy")):
        files.append(f)
        labels.append(class_id - 1)
labels = np.array(labels, dtype=np.int64)


def load_raw(path):
    spec = np.load(path)
    spec = np.ascontiguousarray(spec).astype(np.float32)
    return torch.from_numpy(spec).unsqueeze(0)  # (1, 1024, 365)


# ── Model ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_to_db = T.AmplitudeToDB(top_db=120)


def extract_features(model, transform_name, transform_fn, do_bn_adapt=False):
    """Extract features with a given magnitude transform."""
    # Reset model to pretrained state each time
    model_fresh = dasheng_base_spectrogram().eval().to(device)

    if do_bn_adapt:
        bn_layer = model_fresh.init_bn[1]
        bn_layer.reset_running_stats()
        bn_layer.train()
        with torch.no_grad():
            for start in range(0, len(files), 64):
                batch = [load_raw(f) for f in files[start:start + 64]]
                x = torch.stack(batch)
                if transform_fn is not None:
                    x = transform_fn(x)
                if x.ndim == 3:
                    x = x.unsqueeze(1)
                if x.shape[2] != model_fresh.n_mels:
                    x = torchF.interpolate(
                        x, size=(model_fresh.n_mels, x.shape[3]),
                        mode="bilinear", align_corners=False,
                    )
                with autocast("cuda", enabled=False):
                    model_fresh.init_bn(x.float().to(device))
        bn_layer.eval()

    all_feats = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(files), 16), desc=transform_name):
            batch = [load_raw(f) for f in files[start:start + 16]]
            x = torch.stack(batch)
            if transform_fn is not None:
                x = transform_fn(x)
            out = model_fresh(x.to(device))       # (B, N_tokens, 768)
            all_feats.append(out.mean(dim=1).cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def evaluate_knn(features, name):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs, fold_f1s = [], []
    for train_idx, test_idx in kf.split(np.zeros(len(labels)), labels):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=10, metric="cosine")),
        ])
        pipe.fit(features[train_idx], labels[train_idx])
        y_pred = pipe.predict(features[test_idx])
        fold_accs.append(float((y_pred == labels[test_idx]).mean()))
        fold_f1s.append(float(f1_score(labels[test_idx], y_pred, average="macro", zero_division=0)))

    acc_m, acc_s = np.mean(fold_accs) * 100, np.std(fold_accs) * 100
    f1_m, f1_s = np.mean(fold_f1s) * 100, np.std(fold_f1s) * 100
    print(f"  {name:<30s}  acc={acc_m:.2f}% ± {acc_s:.2f}%  F1={f1_m:.2f}% ± {f1_s:.2f}%")


# ── Run all variants ─────────────────────────────────────────────────────────
print("\nDasheng-base on Glasgow — KNN k=10 cosine + StandardScaler\n")

transforms = [
    ("raw (no transform)",       None,                          False),
    ("log1p",                    lambda x: torch.log1p(x.clamp(min=0)), False),
    ("AmplitudeToDB",            amp_to_db,                     False),
    ("log1p + BN-adapt",         lambda x: torch.log1p(x.clamp(min=0)), True),
    ("AmplitudeToDB + BN-adapt", amp_to_db,                     True),
]

for name, fn, bn_adapt in transforms:
    feats = extract_features(None, name, fn, do_bn_adapt=bn_adapt)
    evaluate_knn(feats, name)
