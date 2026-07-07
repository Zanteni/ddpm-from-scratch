"""
DDPM training loop (Algorithm 1, Ho et al., 2020). Each step samples a
random timestep per image, computes the simplified noise-prediction loss,
and updates the model via backprop.
"""

import torch
import time
from ddpm.diffusion.losses import compute_loss
from ddpm.utils.checkpoint import save_checkpoint


def train_step(model, optimizer, x_0, timesteps, sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod, device="cpu"):
    x_0 = x_0.to(device)
    B = x_0.shape[0]
    t = torch.randint(0, timesteps, (B,), device=device)
    optimizer.zero_grad()
    loss = compute_loss(model=model, x_0=x_0, t=t,
                         sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                         sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, dataloader, optimizer, timesteps, sqrt_alphas_cumprod,
          sqrt_one_minus_alphas_cumprod, num_epochs=10, device="cpu", log_every=100,
          checkpoint_path=None, scheduler=None):
    """
    If scheduler is given (e.g. torch.optim.lr_scheduler.CosineAnnealingLR),
    it's stepped once per epoch, after all batches complete.
    """
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_idx, (x_0, _) in enumerate(dataloader):
            loss = train_step(model, optimizer, x_0, timesteps,
                               sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
            epoch_losses.append(loss)

            if batch_idx % log_every == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss:.4f} | Time: {time.strftime('%H:%M:%S')}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{num_epochs} | avg loss {avg_loss:.4f} | lr {current_lr:.2e} | Time: {time.strftime('%H:%M:%S')}")

        if scheduler is not None:
            scheduler.step()

        if checkpoint_path is not None:
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
            print(f"Saved checkpoint after epoch {epoch}")

    return model