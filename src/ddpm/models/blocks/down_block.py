import torch.nn as nn
from ddpm.models.blocks.residual_block import ResidualBlock
from ddpm.models.blocks.attention_block import AttentionBlock
from ddpm.models.blocks.downsample import Downsample


class DownBlock(nn.Module):
    """
    One encoder stage of the UNet: a stack of ResidualBlocks (each optionally
    followed by self-attention), then a strided-conv downsample that halves
    spatial resolution. Attention is only enabled at specific resolutions in
    the full UNet (16x16 per the original DDPM paper) — most DownBlocks run
    with use_attention=False and use Identity in place of real attention.

    Returns both the downsampled output (fed to the next DownBlock) and the
    list of pre-downsample feature maps (skip connections), which the
    corresponding UpBlock later concatenates back in at matching resolution.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2, use_attention=False, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.num_groups = num_groups

        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                time_emb_dim=time_emb_dim,
                num_groups=num_groups,
            )
            for i in range(num_layers)
        ])

        if self.use_attention:
            self.attn_blocks = nn.ModuleList([
                AttentionBlock(out_channels, num_groups=num_groups)
                for _ in range(num_layers)
            ])
        else:
            self.attn_blocks = nn.ModuleList([
                nn.Identity() for _ in range(num_layers)
            ])

        self.downsample = Downsample(channels=out_channels)

    def forward(self, x, t_emb):
        skip_connections = []

        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, t_emb)
            x = attn_block(x)
            skip_connections.append(x)

        x = self.downsample(x)

        return x, skip_connections