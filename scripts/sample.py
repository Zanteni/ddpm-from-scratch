import torch
import torchvision
from ddpm.models.unet import UNet
from ddpm.diffusion.schedules import get_beta_schedule, get_diffusion_constants
from ddpm.diffusion.sampler import sample
from ddpm.datasets.transforms import unnormalize_to_zero_to_one
from ddpm.utils.checkpoint import load_checkpoint


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    timesteps = 1000
    betas = get_beta_schedule("linear", timesteps)
    consts = get_diffusion_constants(betas)

    model = UNet(
        in_channels=3, base_channels=128, channel_mults=(1, 2, 4),
        time_emb_dim=512, num_layers=2, num_groups=32,
        image_size=32, attn_resolution=16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)  # needed for load_checkpoint's signature

    epoch = load_checkpoint(model, optimizer, None, path="checkpoints/final.pt", device=device)
    print(f"Loaded checkpoint from epoch {epoch}")

    model.to(device)
    model.eval()

    num_samples = 16
    generated = sample(
        model, shape=(num_samples, 3, 32, 32), timesteps=timesteps,
        betas=consts["betas"].to(device), alphas=consts["alphas"].to(device),
        alphas_cumprod=consts["alphas_cumprod"].to(device),
        posterior_variance=consts["posterior_variance"].to(device),
        device=device,
    )

    generated = unnormalize_to_zero_to_one(generated)
    generated = generated.clamp(0, 1)  # guard against tiny numerical overshoot past [0,1]

    torchvision.utils.save_image(generated, "generated_samples.png", nrow=4)
    print("Saved generated_samples.png")


if __name__ == "__main__":
    main()