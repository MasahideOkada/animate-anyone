from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class PoseGuider(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        guiding_embedding_channels: int,
        guiding_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 64, 128),
        activation: str = "relu",
    ):
        super().__init__()

        self.conv_in = gaussian_module(
            nn.Conv2d(guiding_channels, block_out_channels[0], kernel_size=3, padding=1)
        )

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(gaussian_module(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            ))
            self.blocks.append(gaussian_module(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
            ))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], guiding_embedding_channels, kernel_size=3, padding=1)
        )

        match activation.lower():
            case "relu":
                self.act_fn = F.relu
            case "silu" | "swish":
                self.act_fn = F.silu
            case _:
                raise NotImplementedError("`activation` must be `relu`, `silu` or `swish`")

    def forward(self, guiding: torch.FloatTensor, do_normalize: bool = False) -> torch.FloatTensor:
        if do_normalize:
            guiding = 2.0 * guiding - 1.0
        embedding = self.conv_in(guiding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = self.act_fn(embedding)

        embedding = self.conv_out(embedding)
        return embedding

def gaussian_module(module: nn.Module, mean: float = 0.0, std: float = 1.0) -> nn.Module:
    for p in module.parameters():
        nn.init.normal_(p, mean, std)
    return module

def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
