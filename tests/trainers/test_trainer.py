import torch
from torch.utils.data import TensorDataset, DataLoader
from ddpm.diffusion.schedules import get_beta_schedule, get_diffusion_constants
from ddpm.trainers.trainer import train, train_step
from ddpm.models.unet import UNet


def test_train_step_reduces_loss_over_many_steps():
    # Sanity check: repeatedly training on the SAME small batch should
    # cause loss to trend downward, proving the optimizer is actually
    # updating weights in a useful direction.
    B, C, H, W = 4, 3, 16, 16
    T = 100
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    x_0 = torch.randn(B, C, H, W)

    losses = []
    for _ in range(50):
        loss = train_step(model, optimizer, x_0, T,
                           consts["sqrt_alphas_cumprod"], consts["sqrt_one_minus_alphas_cumprod"])
        losses.append(loss)

    print(f"First loss: {losses[0]:.4f}, Last loss: {losses[-1]:.4f}")
    avg_first_10 = sum(losses[:10]) / 10
    avg_last_10 = sum(losses[-10:]) / 10
    print(f"Avg first 10: {avg_first_10:.4f}, Avg last 10: {avg_last_10:.4f}")
    assert avg_last_10 < avg_first_10, "Loss should decrease as the model overfits this single batch"
    print("[OK] Loss decreases over repeated training steps (optimizer is working)")


def test_full_training_loop_runs_with_dummy_dataloader():
    # Use dummy random "images" instead of real CIFAR-10, wrapped in a
    # real DataLoader, to verify the full train() loop runs end-to-end.
    B, C, H, W = 4, 3, 16, 16
    T = 50
    betas = get_beta_schedule("linear", T)
    consts = get_diffusion_constants(betas)

    model = UNet(in_channels=C, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                 num_layers=1, num_groups=4, image_size=H, attn_resolution=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    dummy_images = torch.randn(20, C, H, W)
    dummy_labels = torch.zeros(20, dtype=torch.long)
    dataset = TensorDataset(dummy_images, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    trained_model = train(model, dataloader, optimizer, T,
                           consts["sqrt_alphas_cumprod"], consts["sqrt_one_minus_alphas_cumprod"],
                           num_epochs=2, log_every=1)

    assert trained_model is model
    print("[OK] Full training loop runs end-to-end with a real DataLoader")


if __name__ == "__main__":
    print("===== TRAINER TEST =====")
    test_train_step_reduces_loss_over_many_steps()
    test_full_training_loop_runs_with_dummy_dataloader()
    print("ALL TESTS PASSED")