import torch
from ddpm.datasets.transforms import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one


def test_normalize_boundary_values():
    x = torch.tensor([0.0, 0.5, 1.0])
    out = normalize_to_neg_one_to_one(x)
    assert torch.allclose(out, torch.tensor([-1.0, 0.0, 1.0]))
    print("[OK] normalize_to_neg_one_to_one maps [0, 0.5, 1] -> [-1, 0, 1]")


def test_round_trip_is_identity():
    x = torch.rand(10)  # random values in [0, 1]
    out = unnormalize_to_zero_to_one(normalize_to_neg_one_to_one(x))
    assert torch.allclose(out, x, atol=1e-6)
    print("[OK] normalize -> unnormalize returns the original values")


if __name__ == "__main__":
    print("===== TRANSFORMS TEST =====")
    test_normalize_boundary_values()
    test_round_trip_is_identity()
    print("ALL TESTS PASSED")