import numpy as np
import torch
import torch.nn.functional as F
import timm

from .base import FeatureExtractor


class AudioMAEExtractor(FeatureExtractor):
    """AudioMAE ViT-B/16 pretrained on AudioSet-2M (Huang et al., NeurIPS 2022).

    Architecture: same ViT-B/16 as ImageNet MAE but with single-channel
    spectrogram input (1024 time frames × 128 mel bins → 512 patches).

    The timm key names match the official checkpoint exactly, so weights
    load with strict=False (decoder keys in the checkpoint are ignored).

    Datasets provide raw linear-scale spectrograms (B, 1, F, T).  This
    wrapper applies log1p compression, transposes to (B, 1, T, F) to
    match AudioMAE's training orientation (time = height, frequency =
    width), then resizes to (1024, 128).

    Args:
        path: Path to the downloaded pretrained.pth checkpoint.
    """

    def __init__(self, path: str, device: str = "cpu") -> None:
        self._device = torch.device(device)

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            in_chans=1,
            img_size=(1024, 128),
        )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # Decoder keys in the checkpoint are expected to be unexpected here.
        encoder_missing = [k for k in missing if not k.startswith("head")]
        if encoder_missing:
            raise RuntimeError(f"Missing encoder keys: {encoder_missing}")

        self.model.eval().to(self._device)

    @property
    def name(self) -> str:
        return "audiomae"

    @property
    def embed_dim(self) -> int:
        return 768

    def to(self, device) -> "AudioMAEExtractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Raw linear-scale spectrogram, (B, F, T) or (B, 1, F, T).

        Returns:
            Mean-pooled patch features, shape (B, 768), float32 numpy array.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)                           # (B, 1, F, T)

        # Log compression (reduces dynamic range before standardisation)
        x = torch.log1p(x.clamp(min=0))

        # Transpose to AudioMAE's (time, frequency) convention
        x = x.transpose(2, 3)                            # (B, 1, T, F)

        x = F.interpolate(                               # (B, 1, 1024, 128)
            x.float(), size=(1024, 128),
            mode="bilinear", align_corners=False,
        ).to(self._device)

        # Per-sample standardisation
        b = x.shape[0]
        mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        std  = x.view(b, -1).std(dim=1).view(b, 1, 1, 1).clamp(min=1e-6)
        x = (x - mean) / std

        # Run through timm ViT blocks directly to get all patch tokens
        x = self.model.patch_embed(x)                    # (B, 512, 768)
        cls = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)                  # (B, 513, 768)
        x = self.model.pos_drop(x + self.model.pos_embed)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)                           # (B, 513, 768)

        return x[:, 1:].mean(dim=1).cpu().numpy()        # (B, 768) patch mean, no CLS
