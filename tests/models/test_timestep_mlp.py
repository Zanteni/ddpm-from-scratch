import torch
from ddpm.models.embeddings.timestep_mlp import TimestepMLP


def test_output_shape():
    B = 8
    embedding_dim, hidden_dim, out_dim = 64, 128, 256
    mlp = TimestepMLP(embedding_dim, hidden_dim, out_dim)
    t = torch.randint(0, 1000, (B,))
    out = mlp(t)
    assert out.shape == (B, out_dim)
    print(f"[OK] TimestepMLP output shape correct: {out.shape}")


def test_different_timesteps_produce_different_outputs():
    embedding_dim, hidden_dim, out_dim = 64, 128, 256
    mlp = TimestepMLP(embedding_dim, hidden_dim, out_dim)
    t = torch.tensor([0, 500, 999])
    out = mlp(t)
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])
    print("[OK] Different timesteps produce different MLP outputs")


def test_gradients_flow():
    embedding_dim, hidden_dim, out_dim = 32, 64, 128
    mlp = TimestepMLP(embedding_dim, hidden_dim, out_dim)
    t = torch.randint(0, 1000, (4,))
    out = mlp(t)
    loss = out.sum()
    loss.backward()

    assert mlp.linear1.weight.grad is not None, "linear1 weight has no gradient"
    assert mlp.linear2.weight.grad is not None, "linear2 weight has no gradient"
    assert not torch.all(mlp.linear1.weight.grad == 0), "linear1 gradient is all zeros"
    print("[OK] Gradients flow through linear1 and linear2")


if __name__ == "__main__":
    print("===== TIMESTEP MLP TEST =====")
    test_output_shape()
    test_different_timesteps_produce_different_outputs()
    test_gradients_flow()
    print("ALL TESTS PASSED")