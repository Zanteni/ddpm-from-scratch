import torch
from ddpm.diffusion.schedules import get_beta_schedule, get_diffusion_constants
from ddpm.diffusion.forward_process import forward_diffusion_sample


def test_output_shapes():
    B, C, H, W = 4, 3, 32, 32
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    x_0 = torch.randn(B, C, H, W)
    t = torch.randint(0, T, (B,))

    x_t, eps = forward_diffusion_sample(x_0, t, consts["sqrt_alphas_cumprod"],
                                         consts["sqrt_one_minus_alphas_cumprod"])

    assert x_t.shape == x_0.shape
    assert eps.shape == x_0.shape
    print(f"x_0 shape : {x_0.shape}")
    print(f"x_t shape : {x_t.shape}")
    print("[OK] Output shapes match input")


def test_small_t_keeps_image_close_to_original():
    # At t=0 (or very close to it), almost no noise should have been added,
    # so x_t should be very close to x_0.
    B, C, H, W = 2, 3, 16, 16
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    x_0 = torch.randn(B, C, H, W)
    t = torch.zeros(B, dtype=torch.long)  # t=0 for every sample

    x_t, eps = forward_diffusion_sample(x_0, t, consts["sqrt_alphas_cumprod"],
                                         consts["sqrt_one_minus_alphas_cumprod"])

    diff = (x_t - x_0).abs().mean().item()
    print(f"Mean absolute difference at t=0: {diff:.4f} (should be small)")
    assert diff < 0.1
    print("[OK] At t=0, x_t stays close to x_0")


def test_large_t_looks_like_pure_noise():
    # At t=T-1 (the last timestep), x_t should be dominated by noise,
    # bearing little resemblance to x_0.
    B, C, H, W = 2, 3, 16, 16
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    x_0 = torch.ones(B, C, H, W) * 5.0  # a very distinctive, non-noise-like signal
    t = torch.full((B,), T - 1, dtype=torch.long)

    x_t, eps = forward_diffusion_sample(x_0, t, consts["sqrt_alphas_cumprod"],
                                         consts["sqrt_one_minus_alphas_cumprod"])

    # x_t should be much closer to eps (pure noise) than to x_0 (all 5.0s)
    dist_to_original = (x_t - x_0).abs().mean().item()
    dist_to_noise = (x_t - eps).abs().mean().item()
    print(f"Distance to original signal: {dist_to_original:.4f}")
    print(f"Distance to pure noise      : {dist_to_noise:.4f}")
    assert dist_to_noise < dist_to_original
    print("[OK] At t=T-1, x_t resembles noise much more than the original signal")


def test_different_t_per_sample_in_batch():
    # Confirms t can genuinely differ per sample in the batch, and that
    # this produces different noise levels for each sample independently.
    B, C, H, W = 2, 3, 16, 16
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    x_0 = torch.randn(B, C, H, W)
    t = torch.tensor([0, T - 1])  # sample 0: almost no noise, sample 1: nearly all noise

    x_t, eps = forward_diffusion_sample(x_0, t, consts["sqrt_alphas_cumprod"],
                                         consts["sqrt_one_minus_alphas_cumprod"])

    diff_sample_0 = (x_t[0] - x_0[0]).abs().mean().item()
    diff_sample_1 = (x_t[1] - x_0[1]).abs().mean().item()
    print(f"Sample 0 (t=0) diff from x_0    : {diff_sample_0:.4f}")
    print(f"Sample 1 (t=T-1) diff from x_0  : {diff_sample_1:.4f}")
    assert diff_sample_1 > diff_sample_0
    print("[OK] Different t values per sample produce correspondingly different noise levels")


def test_reproducible_with_fixed_seed():
    # Sanity check on determinism: same seed should reproduce the same x_t.
    B, C, H, W = 2, 3, 8, 8
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    x_0 = torch.randn(B, C, H, W)
    t = torch.randint(0, T, (B,))

    torch.manual_seed(42)
    x_t_1, _ = forward_diffusion_sample(x_0, t, consts["sqrt_alphas_cumprod"],
                                         consts["sqrt_one_minus_alphas_cumprod"])
    torch.manual_seed(42)
    x_t_2, _ = forward_diffusion_sample(x_0, t, consts["sqrt_alphas_cumprod"],
                                         consts["sqrt_one_minus_alphas_cumprod"])

    assert torch.allclose(x_t_1, x_t_2)
    print("[OK] Same seed reproduces identical x_t")


if __name__ == "__main__":
    print("===== FORWARD PROCESS TEST =====")
    test_output_shapes()
    test_small_t_keeps_image_close_to_original()
    test_large_t_looks_like_pure_noise()
    test_different_t_per_sample_in_batch()
    test_reproducible_with_fixed_seed()
    print("ALL TESTS PASSED")