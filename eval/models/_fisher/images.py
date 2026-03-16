import torch
import torch.nn as nn
import numpy as np

from functools import partial
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from enum import Enum, auto
from einops import rearrange
from omegaconf import II

from .mae import get_2d_sincos_pos_embed_flexible, PatchEmbed_new
from .base import D2vModalityConfig, ModalitySpecificEncoder, get_alibi_bias
from .modules import BlockEncoder, FixedPositionalEncoder


class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


@dataclass
class D2vImageConfig(D2vModalityConfig):
    type: Modality = Modality.IMAGE

    input_size: int = 224
    in_chans: int = 1
    patch_size: int = 16
    embed_dim: int = II('model.embed_dim')

    alibi_dims: int = 2
    alibi_distance: str = "manhattan"

    fixed_positions: bool = True

    transformer_decoder: bool = False
    enc_dec_transformer: bool = False
    target_length: int = 1024
    max_length: int = 128
    max_freq: int = 50

    band_width: int = II('model.band_width')
    flatten: str = 'freq'


class ImageEncoder(ModalitySpecificEncoder):

    modality_cfg: D2vImageConfig

    def __init__(
        self,
        modality_cfg: D2vImageConfig,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task=None,
    ):
        self.patch_size = modality_cfg.patch_size
        self.band_width = modality_cfg.band_width
        self.W = self.band_width // self.patch_size
        self.H = modality_cfg.target_length // self.patch_size

        local_encoder = PatchEmbed_new(
            patch_size=modality_cfg.patch_size,
            in_chans=modality_cfg.in_chans,
            embed_dim=modality_cfg.embed_dim,
            stride=modality_cfg.patch_size,
            flatten=modality_cfg.flatten,
        )

        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if modality_cfg.embed_dim != embed_dim:
            local_encoder = nn.Sequential(
                local_encoder,
                nn.Linear(modality_cfg.embed_dim, embed_dim),
            )

        project_features = nn.Identity()

        max_length = modality_cfg.max_length
        max_freq = modality_cfg.max_freq
        pos_embed = nn.Parameter(
            torch.zeros(1, max_length * max_freq, embed_dim), requires_grad=False
        )
        emb = get_2d_sincos_pos_embed_flexible(
            pos_embed.shape[-1], (max_length, max_freq), cls_token=False,
        )
        pos_embed.data.copy_(
            torch.from_numpy(emb[:max_length * max_freq, :]).float().unsqueeze(0)
        )
        fixed_positional_encoder = (
            FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None
        )

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )

        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        alibi_bias_fn = partial(
            get_alibi_bias,
            alibi_biases=alibi_biases,
            heads=modality_cfg.num_alibi_heads,
            dims=modality_cfg.alibi_dims,
            distance=modality_cfg.alibi_distance,
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=None,
            get_alibi_bias=alibi_bias_fn,
        )

    def reset_parameters(self):
        super().reset_parameters()

    def convert_padding_mask(self, x, padding_mask):
        B, T, F = x.shape
        t_extra, f_extra = T % self.patch_size, F % self.patch_size
        padding_mask = padding_mask[:, :-t_extra, :-f_extra]
        padding_mask = rearrange(
            padding_mask,
            'b (tp p) (fp q) -> b tp fp (p q)',
            p=self.patch_size, q=self.patch_size,
        )
        padding_mask = padding_mask.all(-1)
        if self.modality_cfg.flatten == 'time':
            padding_mask = padding_mask.transpose(-2, -1).flatten(1)
        else:
            padding_mask = padding_mask.flatten(1)
        return padding_mask
