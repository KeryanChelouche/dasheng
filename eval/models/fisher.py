import numpy as np
import torch
import torch.nn.functional as F

from .base import FeatureExtractor
from ._fisher.fisher import FISHER


class FISHERExtractor(FeatureExtractor):
    """FISHER-small pretrained on industrial signals (Jiang et al., 2025).

    Architecture: 12-layer ViT with sub-band front-end (band_width=100 freq
    bins per band, patch_size=16). Outputs CLS tokens per sub-band concatenated:
    shape (B, num_bands * embed_dim) where num_bands = freq_bins // 100.

    Datasets provide raw linear-scale spectrograms.  We apply log1p
    compression followed by normalization:
        log1p(x)  →  (· - 3.0173) / (2 × 2.1532)

    For Glasgow (F=1024): 10 sub-bands → (B, 7680)
    For ESC-50  (F=128):   1 sub-band  → (B, 768)

    Args:
        path: Path to FISHER-small.pt checkpoint.
        name: Model name used for caching and reporting.
    """

    # Normalization statistics from FISHER's training data (log-STFT over industrial signals)
    _NORM_MEAN = 3.017344307886898
    _NORM_STD  = 2.1531635155379805

    def __init__(self, path: str, name: str = "fisher_small", device: str = "cpu", freq_bins: int = None) -> None:
        self._device = torch.device(device)
        self._name = name
        self._freq_bins = freq_bins
        self.model = FISHER.from_pretrained(path)
        self.model.eval().to(self._device)
        self._embed_dim = self.model.cfg.embed_dim  # 768 per sub-band

    @property
    def name(self) -> str:
        return self._name

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def to(self, device) -> "FISHERExtractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Raw linear-scale spectrogram, (B, F, T) or (B, 1, F, T).

        Returns:
            CLS-token features per sub-band concatenated,
            shape (B, num_bands * 768), float32 numpy array.
        """
        if x.ndim == 4:
            x = x.squeeze(1)                              # (B, F, T)

        # Transpose to FISHER (time, freq) convention
        x = x.transpose(1, 2).float()                    # (B, T, F)

        # Log-compression
        x = torch.log1p(x.clamp(min=0))

        # FISHER normalization (industrial-signal log-STFT statistics)
        x = (x - self._NORM_MEAN) / (2.0 * self._NORM_STD)

        # Clamp time axis to 1024 frames (model's positional embedding limit)
        if x.shape[1] > 1024:
            x = x[:, :1024]

        # Resize frequency axis to target number of bins (in-distribution w.r.t. training)
        if self._freq_bins is not None and x.shape[2] != self._freq_bins:
            x = F.interpolate(
                x.unsqueeze(1),                          # (B, 1, T, F)
                size=(x.shape[1], self._freq_bins),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)                                 # (B, T, freq_bins)

        # Pad frequency axis to at least band_width if needed
        band_width = self.model.band_width
        if x.shape[2] < band_width:
            x = F.pad(x, (0, band_width - x.shape[2]))

        x = x.unsqueeze(1).to(self._device)              # (B, 1, T, F)

        return self.model.extract_features(x).cpu().numpy()
