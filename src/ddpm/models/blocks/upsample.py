import torch.nn as nn
from ddpm.nn.conv import Conv2d


class Upsample(nn.Module):
    """
    Doubles spatial resolution via nearest-neighbor interpolation followed
    by a convolution. This avoids the checkerboard artifacts commonly
    produced by learned transposed convolutions, while the conv still lets
    the network refine the upsampled result. Channel count is unchanged.
    """

    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x