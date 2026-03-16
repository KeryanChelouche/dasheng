"""Cross-validation evaluation pipeline."""
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger

from .datasets.base import SpectrogramDataset
from .features import extract_and_cache
from .models.base import FeatureExtractor
from .probes.base import Probe


def run_evaluation(
    model: FeatureExtractor,
    dataset: SpectrogramDataset,
    probes: List[Probe],
    device: torch.device,
    batch_size: int = 16,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run cross-validation for all probes and return a results dict.

    The returned dict is JSON-serialisable::

        {
          "model":     "dasheng_base",
          "dataset":   "glasgow",
          "n_samples": 2081,
          "n_classes": 6,
          "probes": {
            "knn_k10": {
              "fold_accs": [0.84, 0.86, ...],
              "mean_acc":  0.8467,
              "std_acc":   0.0136,
            },
            ...
          }
        }
    """
    model.to(device)
    features, labels = extract_and_cache(
        model, dataset, device, batch_size=batch_size, use_cache=use_cache
    )

    results: Dict[str, Any] = {
        "model": model.name,
        "dataset": dataset.name,
        "n_samples": int(len(labels)),
        "n_classes": dataset.n_classes,
        "probes": {},
    }

    for probe in probes:
        fold_accs: List[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(dataset.cv_splits()):
            X_train, y_train = features[train_idx], labels[train_idx]
            X_test, y_test = features[test_idx], labels[test_idx]

            probe.fit(X_train, y_train)
            acc = probe.score(X_test, y_test)
            fold_accs.append(acc)
            logger.debug(f"  {probe.name} fold {fold_idx + 1}: {acc * 100:.2f}%")

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        logger.info(
            f"[{model.name}] [{dataset.name}] {probe.name}: "
            f"{mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%"
        )
        results["probes"][probe.name] = {
            "fold_accs": [float(a) for a in fold_accs],
            "mean_acc": mean_acc,
            "std_acc": std_acc,
        }

    return results
