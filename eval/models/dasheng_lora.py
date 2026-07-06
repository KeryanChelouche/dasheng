"""Dasheng spectrogram backbone with LoRA adapters for supervised fine-tuning.

Mirrors ``dinov3_lora.py`` for the cross-dataset few-shot pipeline:
LoRA-adapt the Dasheng backbone on a source dataset, then freeze the
backbone and fit a linear probe on target few-shot data.

Compatible with ``cross_few_shot.py``'s interface:
  - ``build_dasheng_lora(n_classes) -> nn.Module`` (exposes ``.fc`` head)
  - After fine-tuning, ``model.fc = nn.Identity()`` makes ``forward(x)``
    return mean-pooled features of shape ``(B, embed_dim)`` for probing.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from peft import LoraConfig, get_peft_model

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dasheng import (
    dasheng_06B_spectrogram,
    dasheng_12B_spectrogram,
    dasheng_base_spectrogram,
)

_VARIANTS = {
    "base": (dasheng_base_spectrogram, 768),
    "06B":  (dasheng_06B_spectrogram, 1280),
    "12B":  (dasheng_12B_spectrogram, 1536),
}


def preprocess_batch(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Pass-through preprocess for Dasheng — backbone handles log-compression
    and frequency-axis resizing internally, so we only ship to device."""
    return x.to(device)


class DashengLoRA(nn.Module):
    """DashengSpectrogram backbone with LoRA adapters and a classifier head.

    Accepts pre-computed spectrograms ``(B, F, T)`` or ``(B, 1, F, T)`` at
    raw linear scale — ``log1p`` is applied internally to match the
    original ``DashengExtractor`` preprocessing.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(x.clamp(min=0))
        out = self.backbone(x)          # (B, N_tokens, D)
        pooled = out.mean(dim=1)        # (B, D)
        return self.fc(pooled)


def build_dasheng_lora(
    n_classes: int,
    variant: str = "base",
    rank: int = 16,
    alpha: int = 32,
    path: Optional[str] = None,
    use_dora: bool = False,
) -> DashengLoRA:
    """Build a Dasheng spectrogram backbone with LoRA (or DoRA) + classifier.

    Args:
        n_classes: Number of output classes for the head.
        variant:   One of "base", "06B", "12B".
        rank:      LoRA rank.
        alpha:     LoRA alpha (scaling = alpha / rank).
        path:      Optional local checkpoint path (defaults to Zenodo download).
        use_dora:  If True, use DoRA (weight-decomposed LoRA). Adds a
                   per-channel magnitude vector per adapted module —
                   slightly more params, slightly slower, often a few
                   points better on transfer tasks.
    """
    if variant not in _VARIANTS:
        raise ValueError(
            f"variant must be one of {list(_VARIANTS)}, got {variant!r}"
        )
    factory, embed_dim = _VARIANTS[variant]
    backbone = factory(path=path)

    # Suffix-match avoids the PatchEmbed.proj (Conv2d) which shares the
    # name "proj" with attention's output projection (Linear).
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["attn.qkv", "attn.proj"],
        lora_dropout=0.05,
        bias="none",
        use_dora=use_dora,
    )
    backbone = get_peft_model(backbone, lora_config)

    trainable, total = 0, 0
    for p in backbone.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    method = "DoRA" if use_dora else "LoRA"
    logger.info(
        f"Dasheng-{variant} {method} (r={rank}): {trainable:,} trainable / "
        f"{total:,} total params ({100 * trainable / total:.2f}%)"
    )

    return DashengLoRA(backbone, embed_dim, n_classes)
