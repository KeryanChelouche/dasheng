import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from .base import FeatureExtractor


class Qwen2AudioExtractor(FeatureExtractor):
    """Qwen2-Audio encoder (Chu et al., 2024).

    Architecture: Whisper-large-v3 encoder (32 Transformer layers,
    d_model=1280) further trained during Qwen2-Audio multimodal training,
    with an additional 2× temporal downsampling (output: 750 tokens for
    3000 input frames instead of Whisper's 1500).

    Standalone encoder weights from Atotti/qwen2-audio-encoder (~1.27 GB),
    extracted from the full Qwen2-Audio-7B model.

    Datasets provide raw linear-scale spectrograms (B, F, T).  We apply
    log1p compression, bilinearly resize to (B, 128, 3000), apply
    per-sample standardisation, then run the encoder.

    Args:
        torch_dtype: Weight dtype (default float32).
        device:      Torch device string.
    """

    _HF_ID = "Atotti/qwen2-audio-encoder"
    _MEL_BINS = 128
    _TIME_FRAMES = 3000

    def __init__(
        self,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._dtype = torch_dtype

        self.encoder = AutoModel.from_pretrained(
            self._HF_ID, torch_dtype=torch_dtype,
        )
        self.encoder.eval().to(self._device)

    @property
    def name(self) -> str:
        return "qwen2_audio"

    @property
    def embed_dim(self) -> int:
        return 1280

    def to(self, device) -> "Qwen2AudioExtractor":
        self._device = torch.device(device)
        self.encoder = self.encoder.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Raw linear-scale spectrogram, (B, F, T) or (B, 1, F, T).

        Returns:
            Mean-pooled encoder features, shape (B, 1280), float32 numpy array.
        """
        if x.ndim == 4:
            x = x.squeeze(1)                              # (B, F, T)

        # Log compression (reduces dynamic range before standardisation)
        x = torch.log1p(x.clamp(min=0))

        x = F.interpolate(                               # (B, 128, 3000)
            x.float().unsqueeze(1),
            size=(self._MEL_BINS, self._TIME_FRAMES),
            mode="bilinear", align_corners=False,
        ).squeeze(1)

        # Per-sample standardisation
        b = x.shape[0]
        mean = x.view(b, -1).mean(dim=1).view(b, 1, 1)
        std  = x.view(b, -1).std(dim=1).view(b, 1, 1).clamp(min=1e-6)
        x = (x - mean) / std

        x = x.to(device=self._device, dtype=self._dtype)

        out = self.encoder(x)                             # (B, 750, 1280)
        return out.last_hidden_state.mean(dim=1).float().cpu().numpy()  # (B, 1280)
