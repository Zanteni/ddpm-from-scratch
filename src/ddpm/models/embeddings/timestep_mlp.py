import torch.nn as nn
import torch.nn.functional as F
from ddpm.nn.linear import Linear
from ddpm.models.embeddings.sinusoidal_time_embedding import SinusoidalTimeEmbedding


class TimestepMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # TODO: create self.sinusoidal (SinusoidalTimeEmbedding)
        self.sinusoidal = SinusoidalTimeEmbedding(dim=embedding_dim)
        # TODO: create self.linear1 (Linear: embedding_dim -> hidden_dim)
        self.linear1 = Linear(in_features=embedding_dim,out_features=hidden_dim)
        # TODO: create self.linear2 (Linear: hidden_dim -> out_dim)
        self.linear2 = Linear(in_features=hidden_dim,out_features=out_dim)

    def forward(self, t):
        # TODO: sinusoidal -> linear1 -> silu -> linear2
        t = self.sinusoidal(t)
        t = self.linear1(t)
        t = F.silu(t)
        t = self.linear2(t)
        return t