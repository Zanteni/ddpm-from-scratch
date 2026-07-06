import torch
from ddpm.datasets.cifar10 import get_cifar10_dataloader


def test_dataloader_produces_correct_shape_and_range():
    dataloader = get_cifar10_dataloader(root="./data", batch_size=8, train=True, download=True)

    images, labels = next(iter(dataloader))

    assert images.shape == (8, 3, 32, 32)
    print(f"Batch shape: {images.shape}")

    assert images.min() >= -1.0 and images.max() <= 1.0
    print(f"Value range: [{images.min().item():.4f}, {images.max().item():.4f}]")

    # Sanity check it's not accidentally still in [0, 1] (i.e., normalization
    # actually ran) — real CIFAR-10 images should have some negative values
    assert images.min() < 0, "Expected negative values after [-1,1] normalization"
    print("[OK] CIFAR-10 dataloader produces correctly shaped, normalized batches")


if __name__ == "__main__":
    print("===== CIFAR10 DATALOADER TEST =====")
    test_dataloader_produces_correct_shape_and_range()
    print("ALL TESTS PASSED")