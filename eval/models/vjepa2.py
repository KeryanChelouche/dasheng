from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import FeatureExtractor

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VJEPA21Extractor(FeatureExtractor):
    """V-JEPA 2.1 ViT-B/16 @ 384 distilled from ViT-G.

    Loads the encoder via ``torch.hub`` (facebookresearch/vjepa2) and
    populates it with the EMA encoder weights from a local checkpoint.

    Datasets provide raw linear-scale spectrograms.  We apply log1p
    compression, resize to 384x384, replicate to 3 channels, per-sample
    min-max normalise to [0, 1], then ImageNet-normalise.
    The input is passed in image mode (T=1) and the patch embeddings
    (576 tokens, 768-d) are mean-pooled.

    Args:
        path:   Path to the local ``.pt`` checkpoint.
        device: Torch device string, e.g. "cuda" or "cpu".
    """

    def __init__(self, path: str, device: str = "cpu") -> None:
        self._device = torch.device(device)
        encoder, _ = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_1_vit_base_384",
            pretrained=False,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {
            k.replace("module.backbone.", ""): v
            for k, v in ckpt["ema_encoder"].items()
        }
        encoder.load_state_dict(state)
        self.model = encoder.eval().to(self._device)

    @property
    def name(self) -> str:
        return "vjepa2.1_vitb"

    @property
    def embed_dim(self) -> int:
        return 768

    def to(self, device) -> "VJEPA21Extractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Raw linear-scale spectrogram, (B, F, T) or (B, 1, F, T).

        Returns:
            Mean-pooled features, shape (B, 768), float32 numpy array.
        """
        x = x.to(self._device)
        if x.ndim == 3:
            x = x.unsqueeze(1)                              # (B, 1, F, T)

        # Log compression (reduces dynamic range before min-max)
        x = torch.log1p(x.clamp(min=0))

        x = F.interpolate(                                  # (B, 1, 384, 384)
            x.float(), size=(384, 384),
            mode="bilinear", align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)                           # (B, 3, 384, 384)

        # Per-sample min-max normalisation to [0, 1]
        b = x.shape[0]
        x_flat = x.view(b, -1)
        lo = x_flat.min(dim=1).values.view(b, 1, 1, 1)
        hi = x_flat.max(dim=1).values.view(b, 1, 1, 1)
        x = (x - lo) / (hi - lo + 1e-8)

        # ImageNet channel normalisation
        mean = _IMAGENET_MEAN.to(self._device)
        std  = _IMAGENET_STD.to(self._device)
        x = (x - mean) / std

        # V-JEPA 2.1 image mode: 5D input with T=1
        x = x.unsqueeze(2)                                  # (B, 3, 1, 384, 384)

        out = self.model(x)                                 # (B, 576, 768)
        return out.mean(dim=1).cpu().numpy()                # (B, 768)
