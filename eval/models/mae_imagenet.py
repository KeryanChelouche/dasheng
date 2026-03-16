import numpy as np
import torch
import torch.nn.functional as F
from transformers import ViTMAEModel

from .base import FeatureExtractor

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MAEImageNetExtractor(FeatureExtractor):
    """ViT-B/16 pretrained with MAE on ImageNet-1k (facebook/vit-mae-base).

    This is the key vision baseline: same MAE objective as Dasheng, but
    pretrained on natural images instead of audio spectrograms.

    Spectrograms are resized to 224×224, replicated to 3 channels,
    normalised per-sample to [0, 1], then passed through ImageNet
    normalisation before the ViT encoder.  Mean-pooled patch embeddings
    (768-d) are returned.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self.model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.model.eval().to(self._device)

    @property
    def name(self) -> str:
        return "mae_imagenet"

    @property
    def embed_dim(self) -> int:
        return 768

    def to(self, device) -> "MAEImageNetExtractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Spectrogram tensor, shape (B, F, T) or (B, 1, F, T).

        Returns:
            Mean-pooled patch features, shape (B, 768), float32 numpy array.
        """
        x = x.to(self._device)
        if x.ndim == 3:
            x = x.unsqueeze(1)                              # (B, 1, F, T)

        x = F.interpolate(                                  # (B, 1, 224, 224)
            x.float(), size=(224, 224),
            mode="bilinear", align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)                           # (B, 3, 224, 224)

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

        out = self.model(pixel_values=x.to(self._device))
        # last_hidden_state: (B, N_patches, 768) — no CLS token in ViTMAE
        return out.last_hidden_state.mean(dim=1).cpu().numpy()  # (B, 768)
