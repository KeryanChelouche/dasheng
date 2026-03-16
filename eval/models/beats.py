import numpy as np
import torch
import torch.nn.functional as F

from .base import FeatureExtractor
from ._beats.BEATs import BEATs, BEATsConfig


class BEATsExtractor(FeatureExtractor):
    """BEATs iter3 pretrained on AudioSet-2M (Chen et al., ICML 2023).

    Architecture: 12-layer Transformer with Conv2d patch embedding
    (patch_size=16), relative position bias, and GRU-gated rel-pos.
    Input convention: (B, 1, T, 128) fbank after patch_embedding.

    We bypass the raw-waveform preprocess() path and directly inject
    spectrograms: datasets produce (B, F, T); we transpose to (B, T, F),
    bilinearly resize F → 128 if needed, apply BEATs normalization
    (mean=15.41663, std=6.55582), then run the patch embedding + encoder.

    Args:
        path: Path to BEATs_iter3_AS2M.pt checkpoint.
    """

    # AudioSet fbank statistics used during BEATs training
    _FBANK_MEAN = 15.41663
    _FBANK_STD  = 6.55582

    def __init__(self, path: str, device: str = "cpu") -> None:
        self._device = torch.device(device)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self._device)
        self._embed_dim = cfg.encoder_embed_dim

    @property
    def name(self) -> str:
        return "beats_iter3"

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def to(self, device) -> "BEATsExtractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: (B, F, T) or (B, 1, F, T) spectrogram, frequency-first.

        Returns:
            Mean-pooled token features, shape (B, 768), float32 numpy array.
        """
        if x.ndim == 4:
            x = x.squeeze(1)                              # (B, F, T)

        # Transpose to BEATs (time, frequency) convention
        x = x.transpose(1, 2).float()                    # (B, T, F)

        B, T, Freq = x.shape
        if Freq != 128:
            x = F.interpolate(
                x.unsqueeze(1),                          # (B, 1, T, F)
                size=(T, 128),
                mode="bilinear", align_corners=False,
            ).squeeze(1)                                  # (B, T, 128)

        # BEATs fbank normalization (AudioSet statistics)
        x = (x - self._FBANK_MEAN) / (2 * self._FBANK_STD)
        x = x.to(self._device)

        # Run encoder directly (bypass preprocess / kaldi fbank)
        fbank = x.unsqueeze(1)                           # (B, 1, T, 128)
        features = self.model.patch_embedding(fbank)     # (B, embed_dim, T//p, 128//p)
        features = features.reshape(B, features.shape[1], -1)  # (B, embed_dim, N)
        features = features.transpose(1, 2)             # (B, N, embed_dim)
        features = self.model.layer_norm(features)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        x_enc, _ = self.model.encoder(features)          # (B, N, encoder_embed_dim)

        return x_enc.mean(dim=1).cpu().numpy()            # (B, 768)
