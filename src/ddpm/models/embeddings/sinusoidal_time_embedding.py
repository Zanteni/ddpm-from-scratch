"""
Sinusoidal timestep embedding, as used in DDPM (Ho et al. 2020), borrowed
directly from the Transformer positional embedding (Vaswani et al. 2017).

Why sinusoidal? The network needs to know *which* diffusion timestep t it's
denoising at. Raw integers (t=1, t=500, t=999) are a bad input to a neural
net (unbounded scale, no smooth notion of "close" timesteps). Sinusoidal
embeddings map each integer t to a vector where nearby timesteps produce
similar vectors, and each frequency band lets the network extract different
resolutions of "how far along in the noising process are we."
"""
import math 
import torch
import torch.nn as nn
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        self.base = base
        self.half_dim = dim // 2
        freq = torch.exp(
            -math.log(self.base)
            * torch.arange(self.half_dim, dtype=torch.float32)
            / self.half_dim
        )
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,)
        Returns:
            emb: (B, dim)
        """
        t = t.float()
        args = t[:, None] * self.freq[None, :]
        emb = torch.cat(
            [torch.sin(args), torch.cos(args)],
            dim=1
        )
        return emb