import torch
from ddpm.models.blocks.downsample import Downsample


def test_spatial_size_is_halved():
    B, C, H, W = 2, 32, 32, 32
    down = Downsample(channels=C)
    x = torch.randn(B, C, H, W)
    out = down(x)
    assert out.shape == (B, C, H // 2, W // 2)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] Spatial dimensions correctly halved")


def test_channels_unchanged():
    B, C, H, W = 2, 64, 16, 16
    down = Downsample(channels=C)
    x = torch.randn(B, C, H, W)
    out = down(x)
    assert out.shape[1] == C
    print("[OK] Channel count unchanged")


def test_gradients_flow():
    down = Downsample(channels=16)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    out = down(x)
    loss = out.sum()
    loss.backward()
    assert down.conv.weight.grad is not None, "conv weight has no gradient"
    assert x.grad is not None, "input x has no gradient"
    print("[OK] Gradients flow through downsample conv")


if __name__ == "__main__":
    print("===== DOWNSAMPLE TEST =====")
    test_spatial_size_is_halved()
    test_channels_unchanged()
    test_gradients_flow()
    print("ALL TESTS PASSED")