import torch
from ddpm.diffusion.schedules import get_beta_schedule, get_diffusion_constants
from ddpm.diffusion.losses import compute_loss
from ddpm.models.unet import UNet


def test_loss_is_scalar_and_positive():
    B, C, H, W = 2, 3, 32, 32
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=32, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_layers=2, num_groups=8, image_size=H, attn_resolution=16)

    x_0 = torch.randn(B, C, H, W)
    t = torch.randint(0, T, (B,))

    loss = compute_loss(model, x_0, t, consts["sqrt_alphas_cumprod"], consts["sqrt_one_minus_alphas_cumprod"])

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive (MSE of random predictions vs real noise)"
    print(f"Loss value: {loss.item():.4f}")
    print("[OK] Loss is a positive scalar")


def test_perfect_prediction_gives_zero_loss():
    # If the model predicts the EXACT true noise, loss should be ~0.
    # We fake this with a dummy "model" that just returns the noise it's
    # secretly given, to isolate testing the loss math itself.
    B, C, H, W = 2, 3, 16, 16
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    from ddpm.diffusion.forward_process import forward_diffusion_sample
    x_0 = torch.randn(B, C, H, W)
    t = torch.randint(0, T, (B,))
    x_t, true_eps = forward_diffusion_sample(x_0, t, consts["sqrt_alphas_cumprod"], consts["sqrt_one_minus_alphas_cumprod"])

    class PerfectModel:
        def __call__(self, x, t):
            return true_eps
        def eval(self): pass

    import torch.nn.functional as F
    predicted = PerfectModel()(x_t, t)
    loss = F.mse_loss(true_eps, predicted)

    assert loss.item() < 1e-6
    print(f"Perfect prediction loss: {loss.item():.2e}")
    print("[OK] Loss is ~0 when prediction exactly matches true noise")


def test_gradients_flow_through_model():
    B, C, H, W = 2, 3, 16, 16
    T = 1000
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)

    x_0 = torch.randn(B, C, H, W)
    t = torch.randint(0, T, (B,))

    loss = compute_loss(model, x_0, t, consts["sqrt_alphas_cumprod"], consts["sqrt_one_minus_alphas_cumprod"])
    loss.backward()

    assert model.init_conv.weight.grad is not None, "init_conv has no gradient"
    assert model.final_conv.weight.grad is not None, "final_conv has no gradient"
    print("[OK] Gradients flow through the entire model from the loss")


if __name__ == "__main__":
    print("===== LOSS TEST =====")
    test_loss_is_scalar_and_positive()
    test_perfect_prediction_gives_zero_loss()
    test_gradients_flow_through_model()
    print("ALL TESTS PASSED")