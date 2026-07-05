import torch
from ddpm.nn.normalization import GroupNorm


def test_output_shape():
    B, C, H, W = 4, 16, 8, 8
    gn = GroupNorm(num_groups=4, num_channels=C)
    x = torch.randn(B, C, H, W)
    out = gn(x)
    assert out.shape == (B, C, H, W)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] Shape correct")


def test_normalized_stats_per_group():
    B, C, H, W = 2, 8, 4, 4
    num_groups = 2
    gn = GroupNorm(num_groups=num_groups, num_channels=C)
    x = torch.randn(B, C, H, W) * 5 + 3

    out = gn(x)

    C_per_group = C // num_groups
    out_reshaped = out.reshape(B, num_groups, C_per_group, H, W)

    mean = out_reshaped.mean(dim=(2, 3, 4))
    var = out_reshaped.var(dim=(2, 3, 4), unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)
    print(f"Per-group mean (should be ~0) : {mean.flatten().tolist()}")
    print(f"Per-group var  (should be ~1) : {var.flatten().tolist()}")
    print("[OK] Normalized stats correct per group")


def test_gamma_beta_default_init():
    C = 16
    gn = GroupNorm(num_groups=4, num_channels=C)
    assert torch.allclose(gn.gamma, torch.ones(C))
    assert torch.allclose(gn.beta, torch.zeros(C))
    print("[OK] gamma initialized to ones, beta initialized to zeros")


def test_independent_of_batch_size():
    C, H, W = 8, 4, 4
    num_groups = 2
    gn = GroupNorm(num_groups=num_groups, num_channels=C)

    x_single = torch.randn(1, C, H, W)
    x_batch = torch.cat([x_single, torch.randn(3, C, H, W)], dim=0)

    out_single = gn(x_single)
    out_batch = gn(x_batch)

    assert torch.allclose(out_single[0], out_batch[0], atol=1e-5)
    print("[OK] Output independent of batch size (batch-size invariance holds)")


def test_gamma_beta_are_learnable_parameters():
    gn = GroupNorm(num_groups=4, num_channels=16)
    param_names = [name for name, _ in gn.named_parameters()]
    assert "gamma" in param_names
    assert "beta" in param_names
    print(f"[OK] Learnable parameters registered : {param_names}")


def test_invalid_num_groups_raises():
    try:
        GroupNorm(num_groups=5, num_channels=16)
        assert False, "Expected AssertionError for non-divisible num_groups"
    except AssertionError:
        pass
    print("[OK] Invalid num_groups correctly raises AssertionError")


def test_mismatched_channels_raises():
    gn = GroupNorm(num_groups=4, num_channels=16)
    x = torch.randn(2, 32, 8, 8)
    try:
        gn(x)
        assert False, "Expected AssertionError for channel mismatch"
    except AssertionError:
        pass
    print("[OK] Mismatched input channels correctly raises AssertionError")

def test_gradients_flow_to_gamma_and_beta():
    gn = GroupNorm(num_groups=4, num_channels=16)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)

    out = gn(x)
    loss = out.sum()
    loss.backward()

    assert gn.gamma.grad is not None, "gamma has no gradient"
    assert gn.beta.grad is not None, "beta has no gradient"
    assert not torch.all(gn.gamma.grad == 0), "gamma gradient is all zeros"
    assert not torch.all(gn.beta.grad == 0), "beta gradient is all zeros"
    assert x.grad is not None, "input x has no gradient"

    print(f"[OK] gamma.grad shape : {gn.gamma.grad.shape}")
    print(f"[OK] beta.grad shape  : {gn.beta.grad.shape}")
    print("[OK] Gradients flow correctly to gamma, beta, and input x")
if __name__ == "__main__":
    print("===== GROUPNORM TEST =====")
    test_output_shape()
    test_normalized_stats_per_group()
    test_gamma_beta_default_init()
    test_independent_of_batch_size()
    test_gamma_beta_are_learnable_parameters()
    test_invalid_num_groups_raises()
    test_mismatched_channels_raises()
    test_gradients_flow_to_gamma_and_beta()
    print("ALL TESTS PASSED")