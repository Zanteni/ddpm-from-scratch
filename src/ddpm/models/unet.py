import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm.nn.conv import Conv2d
from ddpm.nn.normalization import GroupNorm
from ddpm.models.embeddings.timestep_mlp import TimestepMLP
from ddpm.models.blocks.down_block import DownBlock
from ddpm.models.blocks.up_block import UpBlock
from ddpm.models.blocks.residual_block import ResidualBlock
from ddpm.models.blocks.attention_block import AttentionBlock


class UNet(nn.Module):
    """
    Full DDPM UNet (Ho et al., 2020): predicts the noise added to an image
    at a given diffusion timestep. Channel counts follow base_channels *
    channel_mults across resolution levels (each DownBlock/UpBlock pair
    halves/doubles spatial resolution). Self-attention is enabled only at
    the resolution matching attn_resolution (16x16 in the original paper)
    and always in the middle/bottleneck block, following the paper's
    finding that attention is most valuable at coarser resolutions where
    full pairwise attention is both affordable and semantically useful.

    Skip connections from every DownBlock are collected into one flat list
    and consumed (via .pop, LIFO) by the UpBlocks in reverse order, giving
    the decoder direct access to the fine spatial detail lost during
    downsampling.
    """

    def __init__(self, in_channels=3, base_channels=128, channel_mults=(1, 2, 4),
                 time_emb_dim=512, num_layers=2, num_groups=32,
                 image_size=32, attn_resolution=16):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.time_emb_dim = time_emb_dim
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.image_size = image_size
        self.attn_resolution = attn_resolution

        self.time_mlp = TimestepMLP(embedding_dim=base_channels, hidden_dim=time_emb_dim, out_dim=time_emb_dim)
        self.init_conv = Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        resolutions = [image_size // (2 ** i) for i in range(len(channel_mults))]
        channels = [base_channels * mult for mult in channel_mults]

        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            output_resolution = resolutions[i + 1]
            use_attn = (output_resolution == attn_resolution)
            self.down_blocks.append(
                DownBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    time_emb_dim=time_emb_dim,
                    num_layers=num_layers,
                    use_attention=use_attn,
                    num_groups=num_groups,
                )
            )

        mid_channel = channels[-1]
        self.mid_block1 = ResidualBlock(
            in_channels=mid_channel, out_channels=mid_channel,
            time_emb_dim=time_emb_dim, num_groups=num_groups,
        )
        self.mid_attn = AttentionBlock(channels=mid_channel, num_groups=num_groups)
        self.mid_block2 = ResidualBlock(
            in_channels=mid_channel, out_channels=mid_channel,
            time_emb_dim=time_emb_dim, num_groups=num_groups,
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            output_resolution = resolutions[i]
            use_attn = (output_resolution == attn_resolution)
            self.up_blocks.append(
                UpBlock(
                    in_channels=channels[i + 1],
                    out_channels=channels[i],
                    skip_channels=channels[i + 1],
                    time_emb_dim=time_emb_dim,
                    num_layers=num_layers,
                    use_attention=use_attn,
                    num_groups=num_groups,
                )
            )

        self.final_norm = GroupNorm(num_groups=num_groups, num_channels=base_channels)
        self.final_conv = Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skip_connections = []
        for down_block in self.down_blocks:
            x, skips = down_block(x, t_emb)
            skip_connections.extend(skips)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for up_block in self.up_blocks:
            x = up_block(x, t_emb, skip_connections)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x