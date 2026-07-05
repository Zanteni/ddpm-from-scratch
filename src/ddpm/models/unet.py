# src/ddpm/models/unet.py
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        raise NotImplementedError("UNet not implemented yet")