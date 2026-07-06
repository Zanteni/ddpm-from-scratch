import torch
from ddpm.models.unet import UNet


def test_output_shape_matches_input():
    B, C, H, W = 2, 3, 32, 32
    model = UNet(in_channels=C, base_channels=32, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_layers=2, num_groups=8, image_size=H, attn_resolution=16)
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))

    out = model(x, t)

    assert out.shape == (B, C, H, W)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] UNet output shape matches input shape exactly")


def test_different_timesteps_give_different_outputs():
    B, C, H, W = 1, 3, 32, 32
    model = UNet(in_channels=C, base_channels=32, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_layers=2, num_groups=8, image_size=H, attn_resolution=16)
    x = torch.randn(B, C, H, W)
    t1 = torch.tensor([10])
    t2 = torch.tensor([900])

    out1 = model(x, t1)
    out2 = model(x, t2)

    assert not torch.allclose(out1, out2), "Output should differ across timesteps"
    print("[OK] Different timesteps produce different predictions")


def test_gradients_flow_through_entire_network():
    B, C, H, W = 2, 3, 32, 32
    model = UNet(in_channels=C, base_channels=32, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_layers=2, num_groups=8, image_size=H, attn_resolution=16)
    x = torch.randn(B, C, H, W, requires_grad=True)
    t = torch.randint(0, 1000, (B,))

    out = model(x, t)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "input x has no gradient"
    assert model.init_conv.weight.grad is not None, "init_conv has no gradient"
    assert model.final_conv.weight.grad is not None, "final_conv has no gradient"
    assert model.mid_block1.conv1.weight.grad is not None, "mid_block1 has no gradient"
    assert model.down_blocks[0].res_blocks[0].conv1.weight.grad is not None, \
        "first down_block has no gradient"
    assert model.up_blocks[-1].res_blocks[-1].conv1.weight.grad is not None, \
        "last up_block has no gradient"
    print("[OK] Gradients flow through the entire network: init_conv -> down -> mid -> up -> final_conv")


def test_parameter_count_reasonable():
    # Sanity check: total parameter count should be substantial (this is a
    # real conv net) but not absurd for the small test config we're using.
    model = UNet(in_channels=3, base_channels=32, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_layers=2, num_groups=8, image_size=32, attn_resolution=16)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    assert num_params > 100_000, "Suspiciously few parameters for a UNet"
    print("[OK] Parameter count is in a reasonable range")


if __name__ == "__main__":
    print("===== UNET TEST =====")
    test_output_shape_matches_input()
    test_different_timesteps_give_different_outputs()
    test_gradients_flow_through_entire_network()
    test_parameter_count_reasonable()
    print("ALL TESTS PASSED")