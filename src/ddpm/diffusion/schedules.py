"""
Beta/alpha schedules for DDPM (Ho et al., 2020).

The forward diffusion process gradually adds Gaussian noise over T
timesteps according to a variance schedule beta_1, ..., beta_T. From these
betas we derive every other quantity needed throughout training and
sampling: alphas, cumulative alpha products, and the posterior variance
used in the reverse process. Precomputing these once and caching them
avoids recomputing this math at every training step.
"""

import torch


def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear beta schedule from the original DDPM paper: betas increase
    linearly from beta_start to beta_end over T steps.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_constants(betas):
    """
    Given a beta schedule, precompute every derived quantity needed by the
    forward process, loss, and sampler.

    Returns a dict with:
        betas, alphas
        alphas_cumprod:                alpha_bar_t = prod_{s<=t} alpha_s
        alphas_cumprod_prev:           alpha_bar_{t-1} (alpha_bar_{-1} := 1)
        sqrt_alphas_cumprod:           sqrt(alpha_bar_t)     -- used in q(x_t|x_0)
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar_t) -- used in q(x_t|x_0)
        posterior_variance:            variance of q(x_{t-1} | x_t, x_0)
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # alpha_bar_{-1} is defined as 1 (no noise added yet), so alpha_bar_prev
    # is alphas_cumprod shifted forward by one position.
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    # Posterior variance (DDPM paper, Eq. 6-7), derived via Bayes' rule for
    # the true reverse process q(x_{t-1} | x_t, x_0).
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }