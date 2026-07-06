import torch
from ddpm.diffusion.schedules import get_beta_schedule, get_diffusion_constants


def test_beta_schedule_shape_and_range():
    T = 1000
    betas = get_beta_schedule(T)
    assert betas.shape == (T,)
    assert betas[0] < betas[-1]  # should increase
    assert torch.all(betas > 0) and torch.all(betas < 1)
    print(f"betas[0]={betas[0]:.6f}, betas[-1]={betas[-1]:.6f}")
    print("[OK] Beta schedule shape and monotonic increase correct")


def test_alphas_cumprod_monotonically_decreasing():
    T = 1000
    betas = get_beta_schedule(T)
    consts = get_diffusion_constants(betas)
    alphas_cumprod = consts["alphas_cumprod"]

    assert torch.all(alphas_cumprod[1:] <= alphas_cumprod[:-1])
    print("[OK] alphas_cumprod is monotonically decreasing (more noise over time)")


def test_alphas_cumprod_boundary_values():
    T = 1000
    betas = get_beta_schedule(T)
    consts = get_diffusion_constants(betas)
    alphas_cumprod = consts["alphas_cumprod"]

    # At t=0, almost no noise has been added yet -> alphas_cumprod ~ 1
    assert alphas_cumprod[0] > 0.99
    # At t=T, nearly all signal has been replaced by noise -> alphas_cumprod ~ 0
    assert alphas_cumprod[-1] < 0.05
    print(f"alphas_cumprod[0]={alphas_cumprod[0]:.4f} (should be ~1)")
    print(f"alphas_cumprod[-1]={alphas_cumprod[-1]:.4f} (should be ~0)")
    print("[OK] alphas_cumprod boundary values correct")


def test_alphas_cumprod_prev_correctly_shifted():
    T = 10
    betas = get_beta_schedule(T)
    consts = get_diffusion_constants(betas)
    alphas_cumprod = consts["alphas_cumprod"]
    alphas_cumprod_prev = consts["alphas_cumprod_prev"]

    assert alphas_cumprod_prev.shape == alphas_cumprod.shape
    assert alphas_cumprod_prev[0].item() == 1.0
    assert torch.allclose(alphas_cumprod_prev[1:], alphas_cumprod[:-1])
    print("[OK] alphas_cumprod_prev correctly shifted, with 1.0 prepended")


def test_sqrt_relations_consistent():
    T = 1000
    betas = get_beta_schedule(T)
    consts = get_diffusion_constants(betas)

    assert torch.allclose(consts["sqrt_alphas_cumprod"], torch.sqrt(consts["alphas_cumprod"]))
    assert torch.allclose(consts["sqrt_one_minus_alphas_cumprod"], torch.sqrt(1.0 - consts["alphas_cumprod"]))
    print("[OK] sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod match their definitions")


def test_posterior_variance_is_non_negative():
    T = 1000
    betas = get_beta_schedule(T)
    consts = get_diffusion_constants(betas)
    posterior_variance = consts["posterior_variance"]

    assert torch.all(posterior_variance >= 0), "Variance can never be negative"
    print(f"posterior_variance range: [{posterior_variance.min():.2e}, {posterior_variance.max():.2e}]")
    print("[OK] posterior_variance is non-negative everywhere")


if __name__ == "__main__":
    print("===== SCHEDULE TEST =====")
    test_beta_schedule_shape_and_range()
    test_alphas_cumprod_monotonically_decreasing()
    test_alphas_cumprod_boundary_values()
    test_alphas_cumprod_prev_correctly_shifted()
    test_sqrt_relations_consistent()
    test_posterior_variance_is_non_negative()
    print("ALL TESTS PASSED")