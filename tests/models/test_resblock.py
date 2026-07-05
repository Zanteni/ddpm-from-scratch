import torch
from ddpm.models.blocks.residual_block import ResidualBlock


def test_output_shape_same_channels():
    B, C, H, W = 2, 64, 16, 16
    time_emb_dim = 128
    block = ResidualBlock(in_channels=C, out_channels=C, time_emb_dim=time_emb_dim)
    x = torch.randn(B, C, H, W)
    t_emb = torch.randn(B, time_emb_dim)
    out = block(x, t_emb)
    assert out.shape == (B, C, H, W)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] Shape correct when in_channels == out_channels")


def test_output_shape_different_channels():
    B, C_in, C_out, H, W = 2, 64, 128, 16, 16
    time_emb_dim = 128
    block = ResidualBlock(in_channels=C_in, out_channels=C_out, time_emb_dim=time_emb_dim)
    x = torch.randn(B, C_in, H, W)
    t_emb = torch.randn(B, time_emb_dim)
    out = block(x, t_emb)
    assert out.shape == (B, C_out, H, W)
    print(f"[OK] Shape correct when in_channels ({C_in}) != out_channels ({C_out})")


def test_residual_conv_type_matches_channel_change():
    same = ResidualBlock(in_channels=64, out_channels=64, time_emb_dim=128)
    diff = ResidualBlock(in_channels=64, out_channels=128, time_emb_dim=128)
    assert isinstance(same.residual_conv, torch.nn.Identity)
    assert not isinstance(diff.residual_conv, torch.nn.Identity)
    print("[OK] residual_conv is Identity when channels match, Conv2d otherwise")


def test_different_timesteps_give_different_outputs():
    # The whole point of injecting t_emb: same image, different timestep,
    # should produce different output. If this fails, the time injection
    # is silently broken (e.g. wrong shape being broadcast/ignored).
    B, C, H, W = 1, 32, 8, 8
    time_emb_dim = 64
    block = ResidualBlock(in_channels=C, out_channels=C, time_emb_dim=time_emb_dim)
    x = torch.randn(B, C, H, W)
    t_emb_1 = torch.randn(B, time_emb_dim)
    t_emb_2 = torch.randn(B, time_emb_dim)

    out1 = block(x, t_emb_1)
    out2 = block(x, t_emb_2)

    assert not torch.allclose(out1, out2), "Output should differ when timestep embedding differs"
    print("[OK] Different timestep embeddings produce different outputs")


def test_gradients_flow_through_all_paths():
    B, C_in, C_out, H, W = 2, 32, 64, 8, 8
    time_emb_dim = 64
    block = ResidualBlock(in_channels=C_in, out_channels=C_out, time_emb_dim=time_emb_dim)
    x = torch.randn(B, C_in, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)

    out = block(x, t_emb)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "input x has no gradient"
    assert t_emb.grad is not None, "time embedding has no gradient"
    assert block.conv1.weight.grad is not None, "conv1 weight has no gradient"
    assert block.time_proj.weight.grad is not None, "time_proj weight has no gradient"
    assert block.residual_conv.weight.grad is not None, "residual_conv weight has no gradient"
    print("[OK] Gradients flow through x, t_emb, conv1, time_proj, and residual_conv")


if __name__ == "__main__":
    print("===== RESIDUAL BLOCK TEST =====")
    test_output_shape_same_channels()
    test_output_shape_different_channels()
    test_residual_conv_type_matches_channel_change()
    test_different_timesteps_give_different_outputs()
    test_gradients_flow_through_all_paths()
    print("ALL TESTS PASSED")