import torch
from ddpm.diffusion.reverse_process import reverse_diffusion_step


@torch.no_grad()
def sample(model, shape, timesteps, betas, alphas, alphas_cumprod, posterior_variance,
           return_intermediates=False, device="cpu"):
    """
    Generates new images by running the full reverse diffusion process,
    starting from pure noise (DDPM Algorithm 2).

    Args:
        shape: (B, C, H, W) — desired output shape
        timesteps: T, total number of diffusion steps
        return_intermediates: if True, also return every x_t along the
            denoising trajectory (useful for visualization)

    Returns:
        x_0, or (x_0, intermediates) if return_intermediates=True
    """
    B = shape[0]
    x_t = torch.randn(shape, device=device)
    intermediates = [x_t] if return_intermediates else None

    for t in reversed(range(timesteps)):
        t_B = torch.full((B,), t, dtype=torch.long, device=device)
        x_t = reverse_diffusion_step(model, x_t, t_B, betas, alphas, alphas_cumprod, posterior_variance)
        if return_intermediates:
            intermediates.append(x_t)

    if return_intermediates:
        return x_t, intermediates
    return x_t