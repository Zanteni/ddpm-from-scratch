import torch
import torch.nn as nn
from ddpm.nn.attention import MultiHeadAttention
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm = GroupNorm(...)
        self.attn = MultiHeadAttention(channels, num_heads)  # from nn/attention.py

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.attn(x)
        return x + residual