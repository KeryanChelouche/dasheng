import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig

from .base import FeatureExtractor


_VARIANTS = {
    "small":    "openai/whisper-small",
    "large_v3": "openai/whisper-large-v3",
}


class WhisperExtractor(FeatureExtractor):
    """Whisper encoder feature extractor (Radford et al., 2023).

    Supports any Whisper variant. Datasets provide raw linear-scale
    spectrograms (B, F, T).  We apply log1p compression, bilinearly
    resize to (B, num_mel_bins, 3000), apply per-sample standardisation,
    and run the encoder.  The decoder is discarded after loading.

    Variant specs:
        small:    80 mel bins, d_model=768,  12 encoder layers, ~88M params
        large_v3: 128 mel bins, d_model=1280, 32 encoder layers, ~800M params

    Args:
        variant:     One of "small", "large_v3".
        torch_dtype: Weight dtype (default float32; use torch.float16 on GPU
                     to halve memory for large_v3).
        device:      Torch device string.
    """

    _TIME_FRAMES = 3000

    def __init__(
        self,
        variant: str = "large_v3",
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS)}, got {variant!r}")

        self._name = f"whisper_{variant}"
        self._device = torch.device(device)
        self._dtype = torch_dtype

        hf_id = _VARIANTS[variant]
        cfg = WhisperConfig.from_pretrained(hf_id)
        self._mel_bins = cfg.num_mel_bins
        self._embed_dim = cfg.d_model

        model = WhisperModel.from_pretrained(hf_id, torch_dtype=torch_dtype)
        self.encoder = model.encoder
        del model  # free decoder weights

        self.encoder.eval().to(self._device)

    @property
    def name(self) -> str:
        return self._name

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def to(self, device) -> "WhisperExtractor":
        self._device = torch.device(device)
        self.encoder = self.encoder.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Raw linear-scale spectrogram, (B, F, T) or (B, 1, F, T).

        Returns:
            Mean-pooled encoder features, shape (B, embed_dim), float32 numpy array.
        """
        if x.ndim == 4:
            x = x.squeeze(1)                              # (B, F, T)

        # Log compression (reduces dynamic range before standardisation)
        x = torch.log1p(x.clamp(min=0))

        x = F.interpolate(                               # (B, mel_bins, 3000)
            x.float().unsqueeze(1),
            size=(self._mel_bins, self._TIME_FRAMES),
            mode="bilinear", align_corners=False,
        ).squeeze(1)

        # Per-sample standardisation
        b = x.shape[0]
        mean = x.view(b, -1).mean(dim=1).view(b, 1, 1)
        std  = x.view(b, -1).std(dim=1).view(b, 1, 1).clamp(min=1e-6)
        x = (x - mean) / std

        x = x.to(device=self._device, dtype=self._dtype)

        out = self.encoder(input_features=x)              # (B, 1500, d_model)
        return out.last_hidden_state.mean(dim=1).float().cpu().numpy()  # (B, d_model)
