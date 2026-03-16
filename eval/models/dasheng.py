import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

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
        self.model = factory(path=path).eval().to(self._device)

    @property
    def name(self) -> str:
        return self._name

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def to(self, device: torch.device | str) -> "DashengExtractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Spectrogram tensor, shape (B, F, T) or (B, 1, F, T).

        Returns:
            Mean-pooled features, shape (B, embed_dim), float32 numpy array.
        """
        out = self.model(x.to(self._device))  # (B, N_tokens, D)
        return out.mean(dim=1).cpu().numpy()   # (B, D)
