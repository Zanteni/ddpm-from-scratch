import torch
from ddpm.models.blocks.upsample import Upsample


def test_spatial_size_is_doubled():
    B, C, H, W = 2, 32, 16, 16
    up = Upsample(channels=C)
    x = torch.randn(B, C, H, W)
    out = up(x)
    assert out.shape == (B, C, H * 2, W * 2)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] Spatial dimensions correctly doubled")


def test_channels_unchanged():
    B, C, H, W = 2, 64, 8, 8
    up = Upsample(channels=C)
    x = torch.randn(B, C, H, W)
    out = up(x)
    assert out.shape[1] == C
    print("[OK] Channel count unchanged")


def test_gradients_flow():
    up = Upsample(channels=16)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    out = up(x)
    loss = out.sum()
    loss.backward()
    assert up.conv.weight.grad is not None, "conv weight has no gradient"
    assert x.grad is not None, "input x has no gradient"
    print("[OK] Gradients flow through upsample conv")


def test_downsample_upsample_roundtrip_shape():
    # Sanity check: Downsample followed by Upsample should return to the
    # original spatial size (though not the original values, since both
    # operations are lossy/learned).
    from ddpm.models.blocks.downsample import Downsample
    B, C, H, W = 2, 32, 16, 16
    down = Downsample(channels=C)
    up = Upsample(channels=C)
    x = torch.randn(B, C, H, W)
    out = up(down(x))
    assert out.shape == x.shape
    print("[OK] Downsample -> Upsample round-trips to original spatial shape")


if __name__ == "__main__":
    print("===== UPSAMPLE TEST =====")
    test_spatial_size_is_doubled()
    test_channels_unchanged()
    test_gradients_flow()
    test_downsample_upsample_roundtrip_shape()
    print("ALL TESTS PASSED")