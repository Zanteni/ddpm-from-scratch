"""
Model/optimizer checkpointing. Saves and restores everything needed to
resume training exactly where it left off: model weights, optimizer state
(momentum estimates, etc.), and the epoch number.
"""

import torch


def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves model + optimizer state to disk.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Loads model + optimizer state from disk, returns the epoch to resume
    from. `model` and `optimizer` must be the same architecture/class as
    what was originally saved — checkpointing resumes training, it doesn't
    convert between different model or optimizer types.
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["epoch"]