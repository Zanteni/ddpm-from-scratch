import torch
import torch.nn.functional as F
from ddpm.nn.conv import Conv2d


def test_output_shape_no_padding():
    B, C_in, H, W = 2, 3, 8, 8
    C_out, K = 16, 3
    conv = Conv2d(C_in, C_out, kernel_size=K, stride=1, padding=0)
    x = torch.randn(B, C_in, H, W)
    out = conv(x)
    expected_H = (H - K) // 1 + 1
    expected_W = (W - K) // 1 + 1
    assert out.shape == (B, C_out, expected_H, expected_W)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print("[OK] Shape correct (no padding)")


def test_output_shape_with_padding_and_stride():
    B, C_in, H, W = 2, 3, 16, 16
    C_out, K, S, P = 8, 3, 2, 1
    conv = Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=P)
    x = torch.randn(B, C_in, H, W)
    out = conv(x)
    expected_H = (H + 2 * P - K) // S + 1
    expected_W = (W + 2 * P - K) // S + 1
    assert out.shape == (B, C_out, expected_H, expected_W)
    print(f"[OK] Shape correct with padding={P}, stride={S}")


def test_matches_pytorch_conv2d():
    # The real correctness check: same weights/bias, same input,
    # output must match PyTorch's own conv2d numerically.
    B, C_in, H, W = 2, 3, 10, 10
    C_out, K, S, P = 6, 3, 2, 1

    conv = Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=P)
    x = torch.randn(B, C_in, H, W)

    my_out = conv(x)
    ref_out = F.conv2d(x, conv.weight, conv.bias, stride=S, padding=P)

    assert torch.allclose(my_out, ref_out, atol=1e-5), \
        f"Max diff: {(my_out - ref_out).abs().max().item()}"
    print(f"[OK] Output matches F.conv2d (max diff: {(my_out - ref_out).abs().max().item():.2e})")


def test_no_bias():
    conv = Conv2d(3, 8, kernel_size=3, bias=False)
    assert conv.bias is None
    x = torch.randn(1, 3, 8, 8)
    out = conv(x)  # should not crash
    print("[OK] bias=False works correctly, no crash")


def test_gradients_flow():
    conv = Conv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    out = conv(x)
    loss = out.sum()
    loss.backward()

    assert conv.weight.grad is not None, "weight has no gradient"
    assert conv.bias.grad is not None, "bias has no gradient"
    assert not torch.all(conv.weight.grad == 0), "weight gradient is all zeros"
    assert x.grad is not None, "input x has no gradient"
    print("[OK] Gradients flow correctly to weight, bias, and input x")


if __name__ == "__main__":
    print("===== CONV2D TEST =====")
    test_output_shape_no_padding()
    test_output_shape_with_padding_and_stride()
    test_matches_pytorch_conv2d()
    test_no_bias()
    test_gradients_flow()
    print("ALL TESTS PASSED")