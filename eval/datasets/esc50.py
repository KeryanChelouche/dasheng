from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T

from .base import SpectrogramDataset


class ESC50Dataset(SpectrogramDataset):
    """ESC-50 environmental sound classification dataset.

    2000 .wav files, 50 classes, 5 predefined folds.

    Audio is loaded via soundfile (torchaudio backend unavailable in
    this environment), resampled to 16 kHz, and converted to a raw
    linear-power Mel spectrogram.  No log/dB transform is applied —
    each FeatureExtractor handles its own magnitude scaling.

    CV: official fold-based protocol — fold i as test, rest as train.
    """

    MEL_PARAMS = dict(
        sample_rate=16000,
        n_fft=1024,
        hop_length=320,
        n_mels=128,
        f_min=20,
        f_max=8000,
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta_path = self.root / "meta" / "esc50.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"ESC-50 metadata not found: {meta_path}")
        self._meta = pd.read_csv(meta_path)
        self._files: List[Path] = [
            self.root / "audio" / fn for fn in self._meta["filename"]
        ]
        self._labels: np.ndarray = self._meta["target"].values.astype(np.int64)
        self._folds: np.ndarray = self._meta["fold"].values
        self._mel = T.MelSpectrogram(**self.MEL_PARAMS)
        self._resamplers: dict = {}

    @property
    def name(self) -> str:
        return "esc50"

    @property
    def n_classes(self) -> int:
        return 50

    @property
    def class_names(self) -> List[str]:
        return (
            self._meta[["target", "category"]]
            .drop_duplicates()
            .sort_values("target")["category"]
            .tolist()
        )

    def items(self) -> Tuple[List[Path], np.ndarray]:
        return self._files, self._labels

    def load_item(self, path: Path) -> torch.Tensor:
        wav, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(wav)
        if wav.ndim == 2:                        # stereo (T, C) → mono (1, T)
            wav = wav.T.mean(0, keepdim=True)
        else:
            wav = wav.unsqueeze(0)               # (1, T)
        if sr != 16000:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, 16000)
            wav = self._resamplers[sr](wav)
        return self._mel(wav)                    # raw linear power, (1, 128, T_frames)

    def cv_splits(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        all_idx = np.arange(len(self._labels))
        for test_fold in sorted(np.unique(self._folds)):
            test_idx = all_idx[self._folds == test_fold]
            train_idx = all_idx[self._folds != test_fold]
            yield train_idx, test_idx
