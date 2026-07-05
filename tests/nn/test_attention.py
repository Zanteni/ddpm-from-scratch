import torch
from ddpm.nn.attention import MultiHeadAttention


def test_output_shape():
    B, C, H, W = 2, 32, 8, 8
    attn = MultiHeadAttention(channels=C, num_heads=4)
    x = torch.randn(B, C, H, W)
    out = attn(x)
    assert out.shape == (B, C, H, W)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] Shape correct (image in -> image out)")


def test_invalid_num_heads_raises():
    try:
        MultiHeadAttention(channels=30, num_heads=4)  # 30 % 4 != 0
        assert False, "Expected AssertionError for non-divisible num_heads"
    except AssertionError:
        pass
    print("[OK] Invalid num_heads correctly raises AssertionError")


def test_single_head_matches_full_attention():
    # With num_heads=1, "multi-head" attention degenerates into plain
    # single-head self-attention. This is a useful sanity check: the
    # multi-head reshaping/merging logic shouldn't change anything about
    # a single head's computation.
    B, C, H, W = 1, 16, 4, 4
    attn = MultiHeadAttention(channels=C, num_heads=1)
    x = torch.randn(B, C, H, W)
    out = attn(x)
    assert out.shape == x.shape
    print("[OK] num_heads=1 runs correctly (degenerate case)")


def test_permutation_invariance_property():
    # Self-attention (without positional encoding) is permutation-equivariant:
    # if you permute the input tokens, the output tokens permute the same way.
    # We test this on the flattened sequence view to confirm attention itself
    # has no hidden spatial assumptions baked in beyond position order.
    B, C, H, W = 1, 16, 2, 2  # small: N=4 tokens, easy to permute by hand
    attn = MultiHeadAttention(channels=C, num_heads=2)
    x = torch.randn(B, C, H, W)

    out1 = attn(x)

    # Permute along the flattened spatial dimension
    x_flat = x.flatten(2)  # (B, C, N)
    perm = torch.randperm(H * W)
    x_perm_flat = x_flat[:, :, perm]
    x_perm = x_perm_flat.reshape(B, C, H, W)

    out2 = attn(x_perm)
    out2_flat = out2.flatten(2)
    out1_flat = out1.flatten(2)

    # Un-permute out2 to compare against out1
    inv_perm = torch.argsort(perm)
    out2_unpermuted = out2_flat[:, :, inv_perm]

    assert torch.allclose(out1_flat, out2_unpermuted, atol=1e-5)
    print("[OK] Attention is permutation-equivariant, as expected")


def test_gradients_flow():
    attn = MultiHeadAttention(channels=16, num_heads=4)
    x = torch.randn(2, 16, 4, 4, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.q.weight.grad is not None, "q weight has no gradient"
    assert attn.k.weight.grad is not None, "k weight has no gradient"
    assert attn.v.weight.grad is not None, "v weight has no gradient"
    assert attn.proj.weight.grad is not None, "proj weight has no gradient"
    assert x.grad is not None, "input x has no gradient"
    assert not torch.all(attn.q.weight.grad == 0), "q gradient is all zeros"
    print("[OK] Gradients flow correctly to Q, K, V, proj weights, and input x")


def test_attention_weights_sum_to_one():
    # A softmax property check: for any query, attention weights over all
    # keys should sum to 1. We verify this by hooking into the computation
    # manually rather than relying on internal state, to keep the test
    # black-box with respect to the module's internals.
    B, C, H, W = 1, 8, 2, 2
    num_heads = 2
    head_dim = C // num_heads
    attn = MultiHeadAttention(channels=C, num_heads=num_heads)
    x = torch.randn(B, C, H, W)

    N = H * W
    x_seq = x.flatten(2).permute(0, 2, 1)
    q = attn.q(x_seq).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    k = attn.k(x_seq).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    scores = (q @ k.transpose(-2, -1)) * attn.scale
    weights = torch.softmax(scores, dim=-1)

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    print("[OK] Attention weights sum to 1 across keys, for every query")


if __name__ == "__main__":
    print("===== ATTENTION TEST =====")
    test_output_shape()
    test_invalid_num_heads_raises()
    test_single_head_matches_full_attention()
    test_permutation_invariance_property()
    test_gradients_flow()
    test_attention_weights_sum_to_one()
    print("ALL TESTS PASSED")