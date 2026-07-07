"""
DDPM training loop (Algorithm 1, Ho et al., 2020). Each step samples a
random timestep per image, computes the simplified noise-prediction loss,
and updates the model via backprop — no need to iterate through the full
diffusion chain during training, since compute_loss uses the forward
process's closed-form shortcut to jump straight to x_t.
"""

import torch
import time
from ddpm.diffusion.losses import compute_loss
from ddpm.utils.checkpoint import save_checkpoint


def train_step(model, optimizer, x_0, timesteps, sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod, device="cpu"):
    """
    Performs one training step: sample random t, compute loss, backprop,
    update weights. Returns the loss as a plain float, for logging.
    """
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
          checkpoint_path=None):
    """
    Full training loop: iterates over the dataloader for num_epochs,
    calling train_step on every batch. Labels from the dataloader are
    discarded — DDPM here is unconditional, it doesn't use class labels.

    If checkpoint_path is given, saves model + optimizer state after every
    epoch, so long training runs can survive interruptions (e.g. a Kaggle
    session timeout) without losing all progress.
    """
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for batch_idx, (x_0, _) in enumerate(dataloader):
            loss = train_step(model, optimizer, x_0, timesteps,
                               sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)

            if batch_idx % log_every == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss:.4f} | Time: {time.strftime('%H:%M:%S')}")

        if checkpoint_path is not None:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"Saved checkpoint after epoch {epoch}")

    return model