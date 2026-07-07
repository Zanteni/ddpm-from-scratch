import torch
from ddpm.models.unet import UNet
from ddpm.diffusion.schedules import get_beta_schedule, get_diffusion_constants
from ddpm.datasets.cifar10 import get_cifar10_dataloader
from ddpm.trainers.trainer import train
from ddpm.utils.checkpoint import save_checkpoint
from ddpm.utils.seed import set_seed


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Diffusion schedule ---
    timesteps = 1000
    betas = get_beta_schedule("linear", timesteps)
    consts = get_diffusion_constants(betas)
    sqrt_alphas_cumprod = consts["sqrt_alphas_cumprod"].to(device)
    sqrt_one_minus_alphas_cumprod = consts["sqrt_one_minus_alphas_cumprod"].to(device)

    # --- Model ---
    model = UNet(
        in_channels=3, base_channels=128, channel_mults=(1, 2, 4),
        time_emb_dim=512, num_layers=2, num_groups=32,
        image_size=32, attn_resolution=16,
    )

    # --- Data ---
    dataloader = get_cifar10_dataloader(root="./data", batch_size=128, train=True, download=True)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # --- Train ---
    trained_model = train(
        model, dataloader, optimizer, timesteps,
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
        num_epochs=10, device=device, log_every=100,
    )

    # --- Save final checkpoint ---
    save_checkpoint(trained_model, optimizer, epoch=10, path="checkpoints/final.pt")
    print("Training complete, checkpoint saved.")


if __name__ == "__main__":
    main()