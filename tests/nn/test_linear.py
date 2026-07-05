import torch
from ddpm.nn.linear import Linear
def test_linear():
    layer = Linear(4, 8)

    x = torch.randn(2, 4)
    y = layer(x)

    print("Input :", x.shape)
    print("Weight:", layer.weight.shape)
    print("Output:", y.shape)

    assert y.shape == (2, 8)

    loss = y.sum()
    loss.backward()

    assert layer.weight.grad is not None

    print("[OK] Shape correct")
    print("[OK] Backward pass works")
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_linear()