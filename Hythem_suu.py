import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,Dataset

class VarLenDataset(Dataset):
    def __init__(self):
        self.seqs = [torch.randn(5), torch.randn(8), torch.randn(3), torch.randn(6)]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

ds = VarLenDataset()
loader = DataLoader(ds, batch_size=2)

for batch in loader:
    print(batch)