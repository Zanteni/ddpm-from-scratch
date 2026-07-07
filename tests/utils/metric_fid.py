import torch
import numpy as np
from ddpm.utils.metrics import compute_fid


def test_fid_is_near_zero_for_identical_distributions():
    torch.manual_seed(42)
    features = torch.randn(200, 64)  # same distribution used for both

    fid = compute_fid(features, features.clone())

    print(f"FID (identical distributions): {fid:.6f}")
    assert abs(fid) < 1e-3, "FID should be ~0 for identical distributions"
    print("[OK] FID is near zero when comparing a distribution to itself")


def test_fid_is_large_for_very_different_distributions():
    torch.manual_seed(42)
    real = torch.randn(200, 64)                  # mean 0, std 1
    fake = torch.randn(200, 64) * 5 + 20          # mean 20, std 5 — very different

    fid = compute_fid(real, fake)

    print(f"FID (very different distributions): {fid:.2f}")
    assert fid > 100, "FID should be large for clearly different distributions"
    print("[OK] FID is large when distributions clearly differ")


def test_fid_is_symmetric():
    torch.manual_seed(42)
    a = torch.randn(200, 64)
    b = torch.randn(200, 64) + 2

    fid_ab = compute_fid(a, b)
    fid_ba = compute_fid(b, a)

    assert abs(fid_ab - fid_ba) < 1e-3, "FID should be symmetric"
    print(f"[OK] FID(a,b)={fid_ab:.4f} matches FID(b,a)={fid_ba:.4f}")


if __name__ == "__main__":
    print("===== FID TEST =====")
    test_fid_is_near_zero_for_identical_distributions()
    test_fid_is_large_for_very_different_distributions()
    test_fid_is_symmetric()
    print("ALL TESTS PASSED")