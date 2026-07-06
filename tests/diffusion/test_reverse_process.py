import torch
from ddpm.diffusion.schedules import get_beta_schedule, get_diffusion_constants
from ddpm.diffusion.reverse_process import reverse_diffusion_step
from ddpm.models.unet import UNet


def test_output_shape():
    B, C, H, W = 2, 3, 16, 16
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)

    x_t = torch.randn(B, C, H, W)
    t = torch.full((B,), 500, dtype=torch.long)

    x_prev = reverse_diffusion_step(model, x_t, t, consts["betas"], consts["alphas"],
                                     consts["alphas_cumprod"], consts["posterior_variance"])

    assert x_prev.shape == x_t.shape
    print(f"x_t shape    : {x_t.shape}")
    print(f"x_{{t-1}} shape : {x_prev.shape}")
    print("[OK] Output shape matches input")


def test_no_noise_added_at_t_zero():
    # At t=0, the step should be fully deterministic (no randomness added) —
    # running it twice with different random seeds should give IDENTICAL
    # results, since eps is forced to zero.
    B, C, H, W = 2, 3, 8, 8
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)
    model.eval()

    x_t = torch.randn(B, C, H, W)
    t = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        torch.manual_seed(1)
        out1 = reverse_diffusion_step(model, x_t, t, consts["betas"], consts["alphas"],
                                       consts["alphas_cumprod"], consts["posterior_variance"])
        torch.manual_seed(999)  # different seed
        out2 = reverse_diffusion_step(model, x_t, t, consts["betas"], consts["alphas"],
                                       consts["alphas_cumprod"], consts["posterior_variance"])

    assert torch.allclose(out1, out2), "t=0 step should be deterministic (no noise added)"
    print("[OK] At t=0, no noise is added — output is fully deterministic")


def test_noise_added_at_t_greater_than_zero():
    # At t>0, different random seeds SHOULD produce different outputs,
    # since real stochastic noise (sigma_t * z) gets added.
    B, C, H, W = 2, 3, 8, 8
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)
    model.eval()

    x_t = torch.randn(B, C, H, W)
    t = torch.full((B,), 500, dtype=torch.long)

    with torch.no_grad():
        torch.manual_seed(1)
        out1 = reverse_diffusion_step(model, x_t, t, consts["betas"], consts["alphas"],
                                       consts["alphas_cumprod"], consts["posterior_variance"])
        torch.manual_seed(999)
        out2 = reverse_diffusion_step(model, x_t, t, consts["betas"], consts["alphas"],
                                       consts["alphas_cumprod"], consts["posterior_variance"])

    assert not torch.allclose(out1, out2), "t>0 step should be stochastic (noise added)"
    print("[OK] At t>0, noise is added — output varies across random seeds")


def test_gradients_flow():
    B, C, H, W = 2, 3, 8, 8
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)

    x_t = torch.randn(B, C, H, W, requires_grad=True)
    t = torch.full((B,), 500, dtype=torch.long)

    x_prev = reverse_diffusion_step(model, x_t, t, consts["betas"], consts["alphas"],
                                     consts["alphas_cumprod"], consts["posterior_variance"])
    loss = x_prev.sum()
    loss.backward()

    assert x_t.grad is not None, "input x_t has no gradient"
    assert model.init_conv.weight.grad is not None, "model has no gradient"
    print("[OK] Gradients flow through the reverse diffusion step")


if __name__ == "__main__":
    print("===== REVERSE PROCESS TEST =====")
    test_output_shape()
    test_no_noise_added_at_t_zero()
    test_noise_added_at_t_greater_than_zero()
    test_gradients_flow()
    print("ALL TESTS PASSED")