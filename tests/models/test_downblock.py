import torch
from ddpm.models.blocks.down_block import DownBlock


def test_output_shapes_no_attention():
    B, C_in, C_out, H, W = 2, 32, 64, 32, 32
    time_emb_dim = 128
    block = DownBlock(in_channels=C_in, out_channels=C_out, time_emb_dim=time_emb_dim,
                       num_layers=2, use_attention=False)
    x = torch.randn(B, C_in, H, W)
    t_emb = torch.randn(B, time_emb_dim)

    out, skips = block(x, t_emb)

    assert out.shape == (B, C_out, H // 2, W // 2)
    assert len(skips) == 2
    for skip in skips:
        assert skip.shape == (B, C_out, H, W)

    print(f"Input shape       : {x.shape}")
    print(f"Downsampled output: {out.shape}")
    print(f"Num skip connections: {len(skips)}, each shape: {skips[0].shape}")
    print("[OK] Shapes correct without attention")


def test_output_shapes_with_attention():
    B, C_in, C_out, H, W = 2, 32, 64, 16, 16
    time_emb_dim = 128
    block = DownBlock(in_channels=C_in, out_channels=C_out, time_emb_dim=time_emb_dim,
                       num_layers=2, use_attention=True, num_groups=8)
    x = torch.randn(B, C_in, H, W)
    t_emb = torch.randn(B, time_emb_dim)

    out, skips = block(x, t_emb)

    assert out.shape == (B, C_out, H // 2, W // 2)
    assert len(skips) == 2
    print("[OK] Shapes correct with attention enabled")


def test_attention_blocks_are_identity_when_disabled():
    block = DownBlock(in_channels=16, out_channels=32, time_emb_dim=64,
                       num_layers=2, use_attention=False, num_groups=8)
    for attn in block.attn_blocks:
        assert isinstance(attn, torch.nn.Identity)
    print("[OK] attn_blocks are Identity when use_attention=False")


def test_attention_blocks_are_real_when_enabled():
    block = DownBlock(in_channels=16, out_channels=32, time_emb_dim=64,
                       num_layers=2, use_attention=True, num_groups=8)
    from ddpm.models.blocks.attention_block import AttentionBlock
    for attn in block.attn_blocks:
        assert isinstance(attn, AttentionBlock)
    print("[OK] attn_blocks are real AttentionBlocks when use_attention=True")


def test_gradients_flow():
    block = DownBlock(in_channels=16, out_channels=32, time_emb_dim=64, num_layers=2, num_groups=8)
    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    t_emb = torch.randn(2, 64, requires_grad=True)

    out, skips = block(x, t_emb)
    loss = out.sum() + sum(s.sum() for s in skips)
    loss.backward()

    assert x.grad is not None, "input x has no gradient"
    assert t_emb.grad is not None, "time embedding has no gradient"
    assert block.res_blocks[0].conv1.weight.grad is not None, "first res block has no gradient"
    print("[OK] Gradients flow through res_blocks, downsample, and both outputs")


if __name__ == "__main__":
    print("===== DOWNBLOCK TEST =====")
    test_output_shapes_no_attention()
    test_output_shapes_with_attention()
    test_attention_blocks_are_identity_when_disabled()
    test_attention_blocks_are_real_when_enabled()
    test_gradients_flow()
    print("ALL TESTS PASSED")