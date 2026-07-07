"""
Model/optimizer checkpointing. Saves and restores everything needed to
resume training exactly where it left off: model weights, optimizer state
(momentum estimates, etc.), and the epoch number.
"""

import torch


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Saves model + optimizer state to disk.
    """
    model_state = (
        model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel)
        else model.state_dict()
    )

    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path, device="cpu"):
    """
    Loads model + optimizer state from disk, returns the epoch to resume
    from. `model` and `optimizer` must be the same architecture/class as
    what was originally saved — checkpointing resumes training, it doesn't
    convert between different model or optimizer types.
    """

    checkpoint = torch.load(path, map_location=device)

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(
            checkpoint["scheduler_state"]
        )

    return checkpoint["epoch"]