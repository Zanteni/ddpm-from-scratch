"""
Single-step reverse diffusion (Ho et al., 2020, Eq. 11): computes x_{t-1}
from x_t using the model's predicted noise. Unlike the forward process,
there's no closed-form shortcut here — denoising genuinely requires
stepping through t = T, T-1, ..., 0 one at a time.

    mu_theta(x_t, t) = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta)
    x_{t-1} = mu_theta + sigma_t * z,  z ~ N(0, I) if t > 0, else z = 0

The reverse process is itself stochastic at every step except the last:
at t=0 no noise is added, giving a deterministic final image.
"""

import torch


def reverse_diffusion_step(model, x_t, t, betas, alphas, alphas_cumprod, posterior_variance):
    """
    Args:
        model: the trained UNet, predicts noise given (x_t, t)
        x_t: current noisy image, shape (B, C, H, W)
        t: current timestep, shape (B,) — typically the same value across
           the batch, since sampling steps through T, T-1, ..., 0 together
        betas, alphas, alphas_cumprod, posterior_variance: precomputed
           schedule constants, shape (T,)

    Returns:
        x_{t-1}: shape (B, C, H, W)
    """
    eps_pred = model(x_t, t)

    B = t.shape[0]
    betas_t = betas[t].reshape(B, 1, 1, 1)
    alphas_t = alphas[t].reshape(B, 1, 1, 1)
    alphas_cumprod_t = alphas_cumprod[t].reshape(B, 1, 1, 1)
    posterior_variance_t = posterior_variance[t].reshape(B, 1, 1, 1)

    mu_theta = (1.0 / torch.sqrt(alphas_t)) * (
        x_t - (betas_t / torch.sqrt(1 - alphas_cumprod_t)) * eps_pred
    )

    sigma_t = torch.sqrt(posterior_variance_t)
    eps = torch.randn_like(x_t) if t[0].item() > 0 else torch.zeros_like(x_t)

    return mu_theta + sigma_t * eps