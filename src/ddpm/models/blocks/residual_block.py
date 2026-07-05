import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm.nn.conv import Conv2d
from ddpm.nn.dropout import Dropout
from ddpm.nn.normalization import GroupNorm
from ddpm.nn.linear import Linear


class ResidualBlock(nn.Module):
    """
    DDPM residual block: two GroupNorm -> SiLU -> Conv2d stages, with the
    diffusion timestep embedding injected between them (added, broadcast
    across all spatial positions), plus a residual connection. If the
    channel count changes, the residual path uses a 1x1 conv (no spatial
    mixing) to match shapes; otherwise it's a plain identity passthrough.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=32, dropout_p=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.time_proj = Linear(time_emb_dim, out_channels)

        self.norm2 = GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.dropout = Dropout(dropout_p)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t = self.time_proj(t_emb)          # (B, out_channels)
        t = t[:, :, None, None]            # (B, out_channels, 1, 1) for broadcasting
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual