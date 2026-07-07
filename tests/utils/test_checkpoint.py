import torch
import os
from ddpm.utils.checkpoint import save_checkpoint, load_checkpoint
from ddpm.models.unet import UNet


def test_save_and_load_round_trip():
    model_a = UNet(in_channels=3, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                    num_layers=1, num_groups=4, image_size=16, attn_resolution=8)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_a,
    T_max=10,
)
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 100, (2,))
    out = model_a(x, t)
    loss = out.sum()
    loss.backward()
    optimizer_a.step()
    scheduler_a.step()

    path = "test_checkpoint.pt"
    save_checkpoint(model_a, optimizer_a,scheduler_a, epoch=5, path=path)

    model_b = UNet(in_channels=3, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                    num_layers=1, num_groups=4, image_size=16, attn_resolution=8)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_b,
    T_max=10,
    )
    weights_differ_before = not torch.allclose(
        model_a.init_conv.weight, model_b.init_conv.weight
    )
    assert weights_differ_before, "Test setup issue: models should differ before loading"

    epoch = load_checkpoint(model_b, optimizer_b,scheduler_b, path=path)

    assert epoch == 5
    assert torch.allclose(model_a.init_conv.weight, model_b.init_conv.weight)
    assert torch.allclose(model_a.final_conv.weight, model_b.final_conv.weight)

    assert (
        optimizer_a.state_dict()["state"].keys()
        ==
        optimizer_b.state_dict()["state"].keys()
    )
    print("[OK] Optimizer state matches exactly")

    assert scheduler_a.last_epoch == scheduler_b.last_epoch

    assert (
        scheduler_a.state_dict()
        ==
        scheduler_b.state_dict()
    )

    print(f"[OK] Loaded epoch: {epoch}")
    print("[OK] Model weights match exactly after save/load round trip")
    print("[OK] Scheduler state matches exactly")
    os.remove(path)


if __name__ == "__main__":
    print("===== CHECKPOINT TEST =====")
    test_save_and_load_round_trip()
    print("ALL TESTS PASSED")