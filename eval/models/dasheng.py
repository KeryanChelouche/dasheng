import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dasheng import (
    dasheng_base_spectrogram,
    dasheng_06B_spectrogram,
    dasheng_12B_spectrogram,
)

from .base import FeatureExtractor

_VARIANTS = {
    "base": (dasheng_base_spectrogram, 768),
    "06B":  (dasheng_06B_spectrogram, 1280),
    "12B":  (dasheng_12B_spectrogram, 1536),
}


class DashengExtractor(FeatureExtractor):
    """Feature extractor wrapping DashengSpectrogram.

    Accepts pre-computed spectrograms of any frequency resolution and
    time length. Internally bilinear-resizes to the model's expected
    frequency dimension (64 bins) if needed, then applies the
    pretrained BatchNorm before the transformer.

    Datasets provide raw linear-scale spectrograms.  This extractor
    applies ``log1p`` compression before feeding into the transformer.

    Args:
        variant: One of "base", "06B", "12B".
        path:    Path to a local checkpoint. If None, downloads from Zenodo.
        device:  Torch device string, e.g. "cuda" or "cpu".
    """

    def __init__(
        self,
        variant: str = "base",
        path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {list(_VARIANTS)}, got {variant!r}"
            )
        factory, dim = _VARIANTS[variant]
        self._name = f"dasheng_{variant}"
        self._embed_dim = dim
        self._device = torch.device(device)
        self._bn_adapted = False
        self.model = factory(path=path).eval().to(self._device)

    @property
    def name(self) -> str:
        if self._bn_adapted:
            return f"{self._name}_bnadapt"
        return self._name

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def to(self, device: torch.device | str) -> "DashengExtractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    def adapt_bn(
        self,
        dataset: "SpectrogramDataset",
        batch_size: int = 64,
    ) -> "DashengExtractor":
        """Recalibrate ``init_bn`` running statistics on *dataset*.

        Does a single forward pass through the dataset, but only through
        AmplitudeToDB → frequency resize → init_bn (skips the full
        transformer), so it's cheap.  After this call the model's name
        includes ``_bnadapt`` to differentiate cached features.
        """
        bn_layer = self.model.init_bn[1]  # BatchNorm2d inside the Sequential
        bn_layer.reset_running_stats()
        bn_layer.train()

        items, _ = dataset.items()
        with torch.no_grad():
            for start in tqdm(
                range(0, len(items), batch_size),
                desc=f"BN-adapt {self._name}/{dataset.name}",
            ):
                batch_items = items[start : start + batch_size]
                tensors = [dataset.load_item(item) for item in batch_items]
                x = torch.stack(tensors)
                x = torch.log1p(x.clamp(min=0))
                if x.ndim == 3:
                    x = x.unsqueeze(1)
                if x.shape[2] != self.model.n_mels:
                    x = F.interpolate(
                        x,
                        size=(self.model.n_mels, x.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                with autocast("cuda", enabled=False):
                    self.model.init_bn(x.float().to(self._device))

        bn_layer.eval()
        self._bn_adapted = True
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Raw linear-scale spectrogram, shape (B, F, T) or (B, 1, F, T).

        Returns:
            Mean-pooled features, shape (B, embed_dim), float32 numpy array.
        """
        # Log-compression
        x = torch.log1p(x.clamp(min=0))
        out = self.model(x.to(self._device))  # (B, N_tokens, D)
        return out.mean(dim=1).cpu().numpy()   # (B, D)
