import torch
from ddpm.nn.dropout import Dropout


def test_eval_mode_passes_through_unchanged():
    dropout = Dropout(p=0.5)
    dropout.eval()
    x = torch.randn(100, 100)
    out = dropout(x)
    assert torch.equal(out, x)
    print("[OK] Eval mode leaves input unchanged")


def test_p_zero_passes_through_unchanged():
    dropout = Dropout(p=0.0)
    dropout.train()
    x = torch.randn(100, 100)
    out = dropout(x)
    assert torch.equal(out, x)
    print("[OK] p=0 leaves input unchanged even in training mode")


def test_invalid_p_raises():
    for bad_p in [-0.1, 1.1]:
        try:
            Dropout(p=bad_p)
            assert False, f"Expected AssertionError for p={bad_p}"
        except AssertionError:
            pass
    print("[OK] Invalid p values correctly raise AssertionError")


def test_training_mode_drops_approximately_p_fraction():
    # Statistical test: with p=0.3 over a large tensor, roughly 30% of
    # elements should be zeroed. We allow generous tolerance since this
    # is a random process, not an exact guarantee.
    torch.manual_seed(0)
    p = 0.3
    dropout = Dropout(p=p)
    dropout.train()
    x = torch.ones(100_000)
    out = dropout(x)

    fraction_zeroed = (out == 0).float().mean().item()
    print(f"Target drop rate: {p}, Observed drop rate: {fraction_zeroed:.4f}")
    assert abs(fraction_zeroed - p) < 0.01, "Observed drop rate too far from target"
    print("[OK] Training mode drops approximately the correct fraction of elements")


def test_expected_value_preserved_by_scaling():
    # The core idea behind "inverted dropout": average output should be
    # close to the average input, because dropped elements are compensated
    # for by scaling the surviving ones up by 1/(1-p).
    torch.manual_seed(0)
    p = 0.4
    dropout = Dropout(p=p)
    dropout.train()
    x = torch.ones(1_000_000) * 5.0
    out = dropout(x)

    mean_out = out.mean().item()
    print(f"Input mean: 5.0, Output mean (should be ~5.0): {mean_out:.4f}")
    assert abs(mean_out - 5.0) < 0.05
    print("[OK] Expected value preserved via inverted dropout scaling")


def test_gradients_flow():
    dropout = Dropout(p=0.5)
    dropout.train()
    x = torch.randn(10, 10, requires_grad=True)
    out = dropout(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "input x has no gradient"
    print("[OK] Gradients flow through dropout during training")


if __name__ == "__main__":
    print("===== DROPOUT TEST =====")
    test_eval_mode_passes_through_unchanged()
    test_p_zero_passes_through_unchanged()
    test_invalid_p_raises()
    test_training_mode_drops_approximately_p_fraction()
    test_expected_value_preserved_by_scaling()
    test_gradients_flow()
    print("ALL TESTS PASSED")