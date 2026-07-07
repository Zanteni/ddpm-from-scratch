import torch
import numpy as np
import random
from ddpm.utils.seed import set_seed


def test_torch_reproducibility():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)
    print("[OK] torch random values reproducible with same seed")


def test_numpy_reproducibility():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)
    print("[OK] numpy random values reproducible with same seed")


def test_python_random_reproducibility():
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b
    print("[OK] Python random values reproducible with same seed")


def test_different_seeds_give_different_values():
    set_seed(1)
    a = torch.randn(5)
    set_seed(2)
    b = torch.randn(5)
    assert not torch.allclose(a, b)
    print("[OK] Different seeds produce different random values")


if __name__ == "__main__":
    print("===== SEED TEST =====")
    test_torch_reproducibility()
    test_numpy_reproducibility()
    test_python_random_reproducibility()
    test_different_seeds_give_different_values()
    print("ALL TESTS PASSED")