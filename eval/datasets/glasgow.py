from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .base import SpectrogramDataset

CLASS_NAMES = [
    "Walking",
    "Sitting down",
    "Standing up",
    "Picking up object",
    "Drinking",
    "Falling",
]


class GlasgowDataset(SpectrogramDataset):
    """Glasgow micro-Doppler human activity recognition dataset.

    2081 spectrograms, 6 activity classes, stored as .npy files of
    shape (1024, 365), float32, Fortran-order, in class-indexed
    subdirectories (1–6).

    Preprocessing: ascontiguousarray → log1p → (1, 1024, 365) tensor.
    CV: stratified 5-fold (shuffle=True, random_state=42).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._files: List[Path] = []
        self._labels: np.ndarray = np.array([], dtype=np.int64)
        self._build_index()

    def _build_index(self) -> None:
        files, labels = [], []
        for class_id in range(1, 7):
            class_dir = self.root / str(class_id)
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            for f in sorted(class_dir.glob("*.npy")):
                files.append(f)
                labels.append(class_id - 1)  # 0-indexed
        self._files = files
        self._labels = np.array(labels, dtype=np.int64)

    @property
    def name(self) -> str:
        return "glasgow"

    @property
    def n_classes(self) -> int:
        return 6

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES

    def items(self) -> Tuple[List[Path], np.ndarray]:
        return self._files, self._labels

    def load_item(self, path: Path) -> torch.Tensor:
        spec = np.load(path)
        spec = np.ascontiguousarray(spec).astype(np.float32)
        spec = np.log1p(spec)
        return torch.from_numpy(spec).unsqueeze(0)  # (1, 1024, 365)

    def cv_splits(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        yield from kf.split(np.zeros(len(self._labels)), self._labels)
