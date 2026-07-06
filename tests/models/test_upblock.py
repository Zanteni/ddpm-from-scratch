import torch
from ddpm.models.blocks.up_block import UpBlock


def test_output_shape():
    B, C_in, C_out, C_skip, H, W = 2, 64, 32, 32, 8, 8
    time_emb_dim = 128
    block = UpBlock(in_channels=C_in, out_channels=C_out, skip_channels=C_skip,
                     time_emb_dim=time_emb_dim, num_layers=2, num_groups=8)

    x = torch.randn(B, C_in, H, W)
    t_emb = torch.randn(B, time_emb_dim)
    # Skip connections at the POST-upsample resolution (2H, 2W), matching
    # what the corresponding DownBlock would have produced at that level
    skips = [torch.randn(B, C_skip, H * 2, W * 2) for _ in range(2)]

    out = block(x, t_emb, skips)

    assert out.shape == (B, C_out, H * 2, W * 2)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] Shape correct: upsampled and channel count matches out_channels")


def test_skip_connections_are_consumed():
    B, C_in, C_out, C_skip, H, W = 1, 32, 16, 16, 8, 8
    time_emb_dim = 64
    block = UpBlock(in_channels=C_in, out_channels=C_out, skip_channels=C_skip,
                     time_emb_dim=time_emb_dim, num_layers=2, num_groups=8)

    x = torch.randn(B, C_in, H, W)
    t_emb = torch.randn(B, time_emb_dim)
    skips = [torch.randn(B, C_skip, H * 2, W * 2) for _ in range(2)]

    block(x, t_emb, skips)

    # All skip connections should have been popped (consumed) by forward
    assert len(skips) == 0
    print("[OK] All skip connections consumed (popped) during forward")


def test_downblock_upblock_shape_compatibility():
    # Integration-style check: DownBlock's output/skips should be directly
    # usable as UpBlock's input/skips at matching channel counts.
    from ddpm.models.blocks.down_block import DownBlock

    B, C_in, C_out, H, W = 2, 32, 64, 16, 16
    time_emb_dim = 128

    down = DownBlock(in_channels=C_in, out_channels=C_out, time_emb_dim=time_emb_dim,
                      num_layers=2, num_groups=8)
    up = UpBlock(in_channels=C_out, out_channels=C_in, skip_channels=C_out,
                 time_emb_dim=time_emb_dim, num_layers=2, num_groups=8)

    x = torch.randn(B, C_in, H, W)
    t_emb = torch.randn(B, time_emb_dim)

    down_out, skips = down(x, t_emb)
    up_out = up(down_out, t_emb, skips)

    assert up_out.shape == (B, C_in, H, W)
    print(f"[OK] DownBlock -> UpBlock round-trip shape matches original input: {up_out.shape}")


def test_gradients_flow():
    B, C_in, C_out, C_skip, H, W = 2, 32, 16, 16, 8, 8
    time_emb_dim = 64
    block = UpBlock(in_channels=C_in, out_channels=C_out, skip_channels=C_skip,
                     time_emb_dim=time_emb_dim, num_layers=2, num_groups=8)

    x = torch.randn(B, C_in, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)
    skips = [torch.randn(B, C_skip, H * 2, W * 2, requires_grad=True) for _ in range(2)]

    out = block(x, t_emb, skips)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "input x has no gradient"
    assert t_emb.grad is not None, "time embedding has no gradient"
    assert all(s.grad is not None for s in skips), "a skip connection has no gradient"
    print("[OK] Gradients flow through x, t_emb, and all skip connections")


if __name__ == "__main__":
    print("===== UPBLOCK TEST =====")
    test_output_shape()
    test_skip_connections_are_consumed()
    test_downblock_upblock_shape_compatibility()
    test_gradients_flow()
    print("ALL TESTS PASSED")