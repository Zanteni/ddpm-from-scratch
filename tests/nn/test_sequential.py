import torch

from ddpm.nn.linear import Linear
from ddpm.nn.sequential import Sequential


def test_sequential():
    print("===== SEQUENTIAL TEST =====")

    model = Sequential(
        Linear(4, 8),
        Linear(8, 16),
        Linear(16, 32)
    )

    x = torch.randn(2, 4)
    y = model(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    assert y.shape == (2, 32)
    print("[OK] Shape correct")

    loss = y.sum()
    loss.backward()

    for i, layer in enumerate(model.layers):
        assert layer.weight.grad is not None
        print(f"[OK] Layer {i} gradients computed")

    print(f"[OK] Number of layers : {len(model)}")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test_sequential()