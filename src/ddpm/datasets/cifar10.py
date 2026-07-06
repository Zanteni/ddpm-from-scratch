"""
CIFAR-10 dataloader for DDPM training. Images are converted to tensors
in [0, 1] via ToTensor(), then remapped to [-1, 1] to match the
zero-centered Gaussian noise used throughout the diffusion process.
"""

import torch
import torchvision
import torchvision.transforms as T
from ddpm.datasets.transforms import normalize_to_neg_one_to_one


def get_cifar10_dataloader(root="./data", batch_size=128, train=True, download=True):
    """
    Returns a DataLoader over CIFAR-10, images normalized to [-1, 1].
    """
    transform = T.Compose([
        T.ToTensor(),
        normalize_to_neg_one_to_one,
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
    )

    return dataloader