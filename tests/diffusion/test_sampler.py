import torch
from ddpm.diffusion.schedules import get_beta_schedule, get_diffusion_constants
from ddpm.diffusion.sampler import sample
from ddpm.models.unet import UNet


def test_output_shape():
    B, C, H, W = 2, 3, 8, 8
    T = 50  # small T for a fast test — sampling loops T times
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)
    model.eval()

    result = sample(model, shape=(B, C, H, W), timesteps=T,
                     betas=consts["betas"], alphas=consts["alphas"],
                     alphas_cumprod=consts["alphas_cumprod"],
                     posterior_variance=consts["posterior_variance"])

    assert result.shape == (B, C, H, W)
    print(f"Sampled output shape: {result.shape}")
    print("[OK] Sampler produces correctly shaped output")


def test_output_is_deterministic_given_fixed_noise_seed():
    # Full sampling involves randomness at every step (except t=0), so
    # different seeds should give different final images.
    B, C, H, W = 1, 3, 8, 8
    T = 50
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)
    model.eval()

    torch.manual_seed(1)
    out1 = sample(model, shape=(B, C, H, W), timesteps=T,
                  betas=consts["betas"], alphas=consts["alphas"],
                  alphas_cumprod=consts["alphas_cumprod"],
                  posterior_variance=consts["posterior_variance"])

    torch.manual_seed(1)
    out2 = sample(model, shape=(B, C, H, W), timesteps=T,
                  betas=consts["betas"], alphas=consts["alphas"],
                  alphas_cumprod=consts["alphas_cumprod"],
                  posterior_variance=consts["posterior_variance"])

    assert torch.allclose(out1, out2), "Same seed should reproduce identical samples"
    print("[OK] Same random seed reproduces identical sampled output")


def test_no_nan_or_inf_in_output():
    # A common failure mode in diffusion sampling: numerical instability
    # producing NaN/Inf values partway through the chain of T steps.
    B, C, H, W = 2, 3, 8, 8
    T = 50
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)
    model.eval()

    result = sample(model, shape=(B, C, H, W), timesteps=T,
                     betas=consts["betas"], alphas=consts["alphas"],
                     alphas_cumprod=consts["alphas_cumprod"],
                     posterior_variance=consts["posterior_variance"])

    assert not torch.isnan(result).any(), "Output contains NaN values"
    assert not torch.isinf(result).any(), "Output contains Inf values"
    print(f"Output range: [{result.min().item():.4f}, {result.max().item():.4f}]")
    print("[OK] No NaN or Inf values in sampled output")


if __name__ == "__main__":
    print("===== SAMPLER TEST =====")
    test_output_shape()
    test_output_is_deterministic_given_fixed_noise_seed()
    test_no_nan_or_inf_in_output()
    print("ALL TESTS PASSED")