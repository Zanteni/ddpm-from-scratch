import torch
import torch.nn as nn
from ddpm.models.blocks.residual_block import ResidualBlock
from ddpm.models.blocks.attention_block import AttentionBlock
from ddpm.models.blocks.upsample import Upsample


class UpBlock(nn.Module):
    """
    One decoder stage of the UNet: upsamples spatial resolution, then for
    each layer concatenates in a skip connection from the corresponding
    DownBlock before running a ResidualBlock (+ optional attention). Skip
    connections restore fine spatial detail lost during downsampling, since
    concatenation (not addition) lets the decoder use both the compressed,
    semantically-rich features from the bottleneck and the high-resolution
    detail preserved by the encoder at this same spatial size.

    skip_connections must be provided in the same order the matching
    DownBlock produced them; they are consumed via .pop() (LIFO order),
    since the most recently produced skip is the first one needed going
    back up.
    """

    def __init__(self, in_channels, out_channels, skip_channels, time_emb_dim,
                 num_layers=2, use_attention=False, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.time_emb_dim = time_emb_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.num_groups = num_groups

        self.upsample = Upsample(channels=in_channels)

        # Every layer's resblock sees a concatenated input (x + skip), not
        # just the first one — unlike DownBlock, where only the first
        # resblock's channel count differs from the rest.
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=(in_channels if i == 0 else out_channels) + skip_channels,
                out_channels=out_channels,
                time_emb_dim=time_emb_dim,
                num_groups=num_groups,
            )
            for i in range(num_layers)
        ])

        if self.use_attention:
            self.attn_blocks = nn.ModuleList([
                AttentionBlock(channels=out_channels, num_groups=num_groups)
                for _ in range(num_layers)
            ])
        else:
            self.attn_blocks = nn.ModuleList([
                nn.Identity() for _ in range(num_layers)
            ])

    def forward(self, x, t_emb, skip_connections):
        x = self.upsample(x)

        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)  # concatenate along the channel dim
            x = res_block(x, t_emb)
            x = attn_block(x)

        return x