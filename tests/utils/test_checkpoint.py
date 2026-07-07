import torch
import os
from ddpm.utils.checkpoint import save_checkpoint, load_checkpoint
from ddpm.models.unet import UNet


def test_save_and_load_round_trip():
    model_a = UNet(in_channels=3, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                    num_layers=1, num_groups=4, image_size=16, attn_resolution=8)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=2e-4, weight_decay=0.01)

    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 100, (2,))
    out = model_a(x, t)
    loss = out.sum()
    loss.backward()
    optimizer_a.step()

    path = "test_checkpoint.pt"
    save_checkpoint(model_a, optimizer_a, epoch=5, path=path)

    model_b = UNet(in_channels=3, base_channels=16, channel_mults=(1, 2), time_emb_dim=64,
                    num_layers=1, num_groups=4, image_size=16, attn_resolution=8)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=2e-4, weight_decay=0.01)

    weights_differ_before = not torch.allclose(
        model_a.init_conv.weight, model_b.init_conv.weight
    )
    assert weights_differ_before, "Test setup issue: models should differ before loading"

    epoch = load_checkpoint(model_b, optimizer_b, path=path)

    assert epoch == 5
    assert torch.allclose(model_a.init_conv.weight, model_b.init_conv.weight)
    assert torch.allclose(model_a.final_conv.weight, model_b.final_conv.weight)
    print(f"[OK] Loaded epoch: {epoch}")
    print("[OK] Model weights match exactly after save/load round trip")

    os.remove(path)


if __name__ == "__main__":
    print("===== CHECKPOINT TEST =====")
    test_save_and_load_round_trip()
    print("ALL TESTS PASSED")