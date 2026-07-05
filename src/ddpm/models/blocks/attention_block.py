import torch.nn as nn
from ddpm.nn.normalization import GroupNorm
from ddpm.nn.attention import MultiHeadAttention


class AttentionBlock(nn.Module):
    """
    Wraps raw self-attention with GroupNorm and a residual connection,
    following the DDPM UNet architecture (Ho et al., 2020). Normalizing
    before attention keeps the softmax numerically stable; the residual
    connection lets gradients skip straight past attention if it isn't
    useful at a given resolution, and lets the network fall back to pure
    convolutional features when attention adds nothing.
    """

    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_groups = num_groups

        self.norm = GroupNorm(num_channels=channels, num_groups=num_groups)
        self.attn = MultiHeadAttention(channels=channels, num_heads=num_heads)

    def forward(self, x):
        assert x.ndim == 3 or x.ndim == 4, f"the input should be a 3D or 4D. got {x.ndim}."
        if x.ndim == 3:
            x = x.unsqueeze(0)

        residual = x
        x = self.norm(x)
        x = self.attn(x)
        return x + residual