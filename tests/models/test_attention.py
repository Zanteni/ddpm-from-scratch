import torch
from ddpm.models.blocks.attention_block import AttentionBlock


def test_output_shape():
    B, C, H, W = 2, 32, 8, 8
    block = AttentionBlock(channels=C, num_heads=4, num_groups=8)
    x = torch.randn(B, C, H, W)
    out = block(x)
    assert out.shape == (B, C, H, W)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] Shape correct (unchanged, as expected)")


def test_residual_connection_matters():
    # If we zero out the attention module's output entirely, the block's
    # output should equal exactly the residual (input unchanged) — proving
    # the residual path is really there and not accidentally overwritten.
    B, C, H, W = 1, 16, 4, 4
    block = AttentionBlock(channels=C, num_heads=2, num_groups=4)

    # Zero out the attention output projection's weight and bias so
    # self.attn(x) always outputs all zeros regardless of input.
    with torch.no_grad():
        block.attn.proj.weight.zero_()
        block.attn.proj.bias.zero_()

    x = torch.randn(B, C, H, W)
    out = block(x)

    assert torch.allclose(out, x, atol=1e-5), "Residual connection is broken"
    print("[OK] Residual connection correctly preserves input when attention output is zero")


def test_gradients_flow():
    block = AttentionBlock(channels=16, num_heads=4, num_groups=4)
    x = torch.randn(2, 16, 4, 4, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()

    assert block.norm.gamma.grad is not None, "GroupNorm gamma has no gradient"
    assert block.attn.q.weight.grad is not None, "attention Q weight has no gradient"
    assert x.grad is not None, "input x has no gradient"
    print("[OK] Gradients flow through GroupNorm, attention, and residual path")


def test_invalid_group_channel_combo_raises():
    try:
        AttentionBlock(channels=30, num_heads=2, num_groups=32)  # 30 % 32 != 0
        assert False, "Expected AssertionError from GroupNorm"
    except AssertionError:
        pass
    print("[OK] Invalid channels/num_groups combo correctly raises AssertionError")


if __name__ == "__main__":
    print("===== ATTENTION BLOCK TEST =====")
    test_output_shape()
    test_residual_connection_matters()
    test_gradients_flow()
    test_invalid_group_channel_combo_raises()
    print("ALL TESTS PASSED")