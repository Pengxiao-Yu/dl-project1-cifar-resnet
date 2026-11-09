"""Structural and training checks on synthetic inputs (not experiment results)."""

import json, tempfile
from pathlib import Path
import torch
from model import BasicBlock, CifarResNet, parameter_count
from train import augment, atomic_save


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    model = CifarResNet()
    assert parameter_count(model) == 4368690
    block = BasicBlock(8, 8).eval()
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()
    x = torch.randn(2, 8, 5, 5)
    torch.testing.assert_close(block(x), x.relu())
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(4))
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with tempfile.TemporaryDirectory() as tmp:
        model.eval()
        before = model(x).detach()
        atomic_save(model.state_dict(), Path(tmp) / "model.pt")
        other = CifarResNet()
        other.load_state_dict(torch.load(Path(tmp) / "model.pt", weights_only=True))
        other.eval()
        torch.testing.assert_close(before, other(x))
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        # Each image is constant and distinct: augmentation must not exchange examples/channels.
        v = (
            torch.arange(4, dtype=torch.uint8)
            .view(4, 1, 1, 1)
            .expand(4, 3, 32, 32)
            .to(device)
        )
        aug = augment(v)
        torch.testing.assert_close(aug.cpu(), v.cpu().float() / 255)
        assert aug.shape == v.shape
    return {
        "parameters": parameter_count(model),
        "budget_pass": True,
        "residual_identity_pass": True,
        "finite_gradients_pass": True,
        "checkpoint_roundtrip_pass": True,
        "augmentation_no_cross_image_mix_pass": True,
        "synthetic_only": True,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
