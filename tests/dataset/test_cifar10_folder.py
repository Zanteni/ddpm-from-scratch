import torch
import os
from PIL import Image
import numpy as np
from ddpm.datasets.cifar10 import get_cifar10_dataloader_from_folder


def test_dataloader_from_folder_produces_correct_shape_and_range():
    # Build a tiny fake dataset: 2 classes, 3 images each, matching the
    # root/<class_name>/*.png structure that ImageFolder expects.
    root = "test_fake_cifar"
    for class_name in ["cat", "dog"]:
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(3):
            fake_pixels = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(fake_pixels).save(os.path.join(class_dir, f"{i}.png"))

    dataloader = get_cifar10_dataloader_from_folder(root=root, batch_size=2, shuffle=False)
    images, labels = next(iter(dataloader))

    assert images.shape == (2, 3, 32, 32)
    assert images.min() >= -1.0 and images.max() <= 1.0
    assert images.min() < 0, "Expected negative values after [-1,1] normalization"
    print(f"Batch shape: {images.shape}")
    print(f"Value range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print("[OK] ImageFolder-based CIFAR-10 dataloader produces correctly shaped, normalized batches")

    # cleanup
    import shutil
    shutil.rmtree(root)


if __name__ == "__main__":
    print("===== CIFAR10 FOLDER DATALOADER TEST =====")
    test_dataloader_from_folder_produces_correct_shape_and_range()
    print("ALL TESTS PASSED")