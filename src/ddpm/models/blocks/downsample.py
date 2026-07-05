import torch.nn as nn
from ddpm.nn.conv import Conv2d


class Downsample(nn.Module):
    """
    Halves spatial resolution via a stride-2 convolution (the approach used
    in the original DDPM paper), while keeping channel count unchanged.
    Channel transformations happen in ResidualBlock, not here — this layer's
    only job is shrinking H and W.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)