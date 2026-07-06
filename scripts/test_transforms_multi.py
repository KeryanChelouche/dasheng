#!/usr/bin/env python3
"""Test magnitude transforms on Glasgow for Dasheng, BEATs, and FISHER.

For Dasheng: adds raw + BN-adapt to the previous results.
For BEATs/FISHER: tests whether the transform choice matters, and whether
adapting normalization stats on the target domain helps.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torchF
import torchaudio.transforms as T
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── Load Glasgow ─────────────────────────────────────────────────────────────
DATA_ROOT = REPO_ROOT / "data" / "Glasgow"
files, labels = [], []
for class_id in range(1, 7):
    for f in sorted((DATA_ROOT / str(class_id)).glob("*.npy")):
        files.append(f)
        labels.append(class_id - 1)
labels = np.array(labels, dtype=np.int64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_to_db = T.AmplitudeToDB(top_db=120)


def load_raw(path):
    spec = np.load(path)
    return torch.from_numpy(np.ascontiguousarray(spec).astype(np.float32)).unsqueeze(0)


def evaluate_knn(features, name):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    for train_idx, test_idx in kf.split(np.zeros(len(labels)), labels):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=10, metric="cosine")),
        ])
        pipe.fit(features[train_idx], labels[train_idx])
        y_pred = pipe.predict(features[test_idx])
        fold_accs.append(float((y_pred == labels[test_idx]).mean()))
    acc_m, acc_s = np.mean(fold_accs) * 100, np.std(fold_accs) * 100
    print(f"  {name:<40s}  acc={acc_m:.2f}% ± {acc_s:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# DASHENG — raw + BN-adapt
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== DASHENG-BASE on Glasgow ===\n")

from dasheng import dasheng_base_spectrogram


def dasheng_extract(transform_name, transform_fn, do_bn_adapt=False):
    model = dasheng_base_spectrogram().eval().to(device)

    if do_bn_adapt:
        bn = model.init_bn[1]
        bn.reset_running_stats()
        bn.train()
        with torch.no_grad():
            for start in range(0, len(files), 64):
                batch = [load_raw(f) for f in files[start:start + 64]]
                x = torch.stack(batch)
                if transform_fn is not None:
                    x = transform_fn(x)
                if x.ndim == 3:
                    x = x.unsqueeze(1)
                if x.shape[2] != model.n_mels:
                    x = torchF.interpolate(x, size=(model.n_mels, x.shape[3]),
                                           mode="bilinear", align_corners=False)
                with autocast("cuda", enabled=False):
                    model.init_bn(x.float().to(device))
        bn.eval()

    all_feats = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(files), 16), desc=transform_name, leave=False):
            batch = [load_raw(f) for f in files[start:start + 16]]
            x = torch.stack(batch)
            if transform_fn is not None:
                x = transform_fn(x)
            out = model(x.to(device))
            all_feats.append(out.mean(dim=1).cpu().numpy())
    return np.concatenate(all_feats)


for name, fn, bn in [
    ("raw",                   None,                                    False),
    ("raw + BN-adapt",        None,                                    True),
    ("log1p",                 lambda x: torch.log1p(x.clamp(min=0)),   False),
    ("log1p + BN-adapt",      lambda x: torch.log1p(x.clamp(min=0)),   True),
    ("AmplitudeToDB",         amp_to_db,                               False),
    ("AmplitudeToDB + BN-adapt", amp_to_db,                            True),
]:
    feats = dasheng_extract(name, fn, do_bn_adapt=bn)
    evaluate_knn(feats, name)


# ══════════════════════════════════════════════════════════════════════════════
# BEATs — test transforms + stat adaptation
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BEATs iter3 on Glasgow ===\n")

sys.path.insert(0, str(REPO_ROOT / "eval" / "models"))
from _beats.BEATs import BEATs, BEATsConfig

BEATS_PATH = REPO_ROOT / "BEATs_iter3.pt"
if BEATS_PATH.exists():
    def beats_extract(transform_name, transform_fn, adapt_stats=False):
        ckpt = torch.load(str(BEATS_PATH), map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval().to(device)

        # First pass: compute adapted stats if needed
        fbank_mean, fbank_std = 15.41663, 6.55582  # AudioSet defaults
        if adapt_stats:
            all_vals = []
            for start in range(0, len(files), 64):
                batch = [load_raw(f) for f in files[start:start + 64]]
                x = torch.stack(batch)
                if transform_fn is not None:
                    x = transform_fn(x)
                if x.ndim == 4:
                    x = x.squeeze(1)
                x = x.transpose(1, 2).float()  # (B, T, F)
                if x.shape[2] != 128:
                    x = torchF.interpolate(x.unsqueeze(1), size=(x.shape[1], 128),
                                           mode="bilinear", align_corners=False).squeeze(1)
                all_vals.append(x.reshape(-1).numpy())
            all_vals = np.concatenate(all_vals)
            fbank_mean = float(np.mean(all_vals))
            fbank_std = float(np.std(all_vals))

        # Extract features
        all_feats = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(files), 16), desc=transform_name, leave=False):
                batch = [load_raw(f) for f in files[start:start + 16]]
                x = torch.stack(batch)
                if transform_fn is not None:
                    x = transform_fn(x)
                if x.ndim == 4:
                    x = x.squeeze(1)
                x = x.transpose(1, 2).float()
                B, T, F = x.shape
                if F != 128:
                    x = torchF.interpolate(x.unsqueeze(1), size=(T, 128),
                                           mode="bilinear", align_corners=False).squeeze(1)
                x = (x - fbank_mean) / (2 * fbank_std)
                x = x.to(device)
                fbank = x.unsqueeze(1)
                features = model.patch_embedding(fbank)
                features = features.reshape(B, features.shape[1], -1).transpose(1, 2)
                features = model.layer_norm(features)
                if model.post_extract_proj is not None:
                    features = model.post_extract_proj(features)
                x_enc, _ = model.encoder(features)
                all_feats.append(x_enc.mean(dim=1).cpu().numpy())
        return np.concatenate(all_feats)

    for name, fn, adapt in [
        ("log (current)",           lambda x: torch.log(x.clamp(min=1e-10)),   False),
        ("log1p",                   lambda x: torch.log1p(x.clamp(min=0)),     False),
        ("raw",                     None,                                       False),
        ("log + adapt stats",       lambda x: torch.log(x.clamp(min=1e-10)),   True),
        ("log1p + adapt stats",     lambda x: torch.log1p(x.clamp(min=0)),     True),
        ("raw + adapt stats",       None,                                       True),
    ]:
        feats = beats_extract(name, fn, adapt_stats=adapt)
        evaluate_knn(feats, name)
else:
    print("  (BEATs checkpoint not found, skipping)")


# ══════════════════════════════════════════════════════════════════════════════
# FISHER — test transforms + stat adaptation
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== FISHER-small on Glasgow ===\n")

from _fisher.fisher import FISHER

FISHER_PATH = REPO_ROOT / "FISHER-small.pt"
if FISHER_PATH.exists():
    def fisher_extract(transform_name, transform_fn, adapt_stats=False):
        model = FISHER.from_pretrained(str(FISHER_PATH))
        model.eval().to(device)
        band_width = model.band_width

        norm_mean, norm_std = 3.017344307886898, 2.1531635155379805
        if adapt_stats:
            all_vals = []
            for start in range(0, len(files), 64):
                batch = [load_raw(f) for f in files[start:start + 64]]
                x = torch.stack(batch)
                if transform_fn is not None:
                    x = transform_fn(x)
                if x.ndim == 4:
                    x = x.squeeze(1)
                x = x.transpose(1, 2).float()
                if x.shape[1] > 1024:
                    x = x[:, :1024]
                all_vals.append(x.reshape(-1).numpy())
            all_vals = np.concatenate(all_vals)
            norm_mean = float(np.mean(all_vals))
            norm_std = float(np.std(all_vals))

        all_feats = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(files), 16), desc=transform_name, leave=False):
                batch = [load_raw(f) for f in files[start:start + 16]]
                x = torch.stack(batch)
                if transform_fn is not None:
                    x = transform_fn(x)
                if x.ndim == 4:
                    x = x.squeeze(1)
                x = x.transpose(1, 2).float()
                if x.shape[1] > 1024:
                    x = x[:, :1024]
                x = (x - norm_mean) / (2.0 * norm_std)
                if x.shape[2] < band_width:
                    x = torchF.pad(x, (0, band_width - x.shape[2]))
                x = x.unsqueeze(1).to(device)
                all_feats.append(model.extract_features(x).cpu().numpy())
        return np.concatenate(all_feats)

    for name, fn, adapt in [
        ("log (current)",           lambda x: torch.log(x.clamp(min=0) + 1e-10), False),
        ("log1p",                   lambda x: torch.log1p(x.clamp(min=0)),        False),
        ("raw",                     None,                                          False),
        ("log + adapt stats",       lambda x: torch.log(x.clamp(min=0) + 1e-10), True),
        ("log1p + adapt stats",     lambda x: torch.log1p(x.clamp(min=0)),        True),
        ("raw + adapt stats",       None,                                          True),
    ]:
        feats = fisher_extract(name, fn, adapt_stats=adapt)
        evaluate_knn(feats, name)
else:
    print("  (FISHER checkpoint not found, skipping)")
