import torch
from ddpm.models.embeddings.sinusoidal_time_embedding import SinusoidalTimeEmbedding


def test_output_shape():
    B = 8
    dim = 64
    emb_layer = SinusoidalTimeEmbedding(dim)
    t = torch.randint(0, 1000, (B,))
    out = emb_layer(t)
    assert out.shape == (B, dim), f"Expected {(B, dim)}, got {out.shape}"


def test_odd_dim_raises_assertion():
    try:
        SinusoidalTimeEmbedding(dim=63)
        assert False, "Expected AssertionError for odd dim"
    except AssertionError:
        pass


def test_different_timesteps_produce_different_embeddings():
    dim = 64
    emb_layer = SinusoidalTimeEmbedding(dim)
    t = torch.tensor([0, 500, 999])
    out = emb_layer(t)
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])


def test_nearby_timesteps_are_more_similar_than_far_ones():
    dim = 64
    emb_layer = SinusoidalTimeEmbedding(dim)

    t_close = torch.tensor([100, 101])
    t_far = torch.tensor([100, 900])

    out_close = emb_layer(t_close)
    out_far = emb_layer(t_far)

    dist_close = torch.norm(out_close[0] - out_close[1])
    dist_far = torch.norm(out_far[0] - out_far[1])

    assert dist_close < dist_far, "Nearby timesteps should be closer in embedding space"


def test_buffer_moves_with_module():
    # register_buffer should mean `freq` is included in state_dict
    # and moves correctly with .to(device) / .float() etc.
    dim = 32
    emb_layer = SinusoidalTimeEmbedding(dim)
    assert "freq" in dict(emb_layer.named_buffers())
    assert emb_layer.freq.shape == (dim // 2,)


def test_zero_timestep_embedding():
    # At t=0, sin(0)=0 for all frequencies, cos(0)=1 for all frequencies
    dim = 16
    emb_layer = SinusoidalTimeEmbedding(dim)
    t = torch.tensor([0])
    out = emb_layer(t)
    half = dim // 2
    assert torch.allclose(out[0, :half], torch.zeros(half), atol=1e-6)
    assert torch.allclose(out[0, half:], torch.ones(half), atol=1e-6)


if __name__ == "__main__":
    test_output_shape()
    test_odd_dim_raises_assertion()
    test_different_timesteps_produce_different_embeddings()
    test_nearby_timesteps_are_more_similar_than_far_ones()
    test_buffer_moves_with_module()
    test_zero_timestep_embedding()
    print("All time embedding tests passed.")