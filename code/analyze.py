"""Create validation diagnostics from the selected checkpoint, not hidden labels."""

import argparse, json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import CifarResNet

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/competition.npz")
    p.add_argument("--run", default="results/main")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    out = Path(a.run)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    h = json.loads((out / "history.json").read_text())
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    data = np.load(a.data)
    va = data["val_indices"]
    images = data["train_images"][va]
    y = data["train_labels"][va]
    m = CifarResNet(c["width"]).to(a.device)
    m.load_state_dict(ck["model"])
    m.eval()
    mean = torch.tensor(c["mean"], device=a.device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    std = torch.tensor(c["std"], device=a.device, dtype=torch.float32).view(1, 3, 1, 1)
    pr = []
    for i in range(0, len(y), 256):
        x = (
            torch.tensor(
                images[i : i + 256].transpose(0, 3, 1, 2).copy(),
                device=a.device,
                dtype=torch.float32,
            )
            / 255
        )
        pr.extend(m((x - mean) / std).argmax(1).cpu().tolist())
    pr = np.array(pr)
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, pr), 1)
    np.save(out / "validation_predictions.npy", pr)
    np.save(out / "confusion.npy", cm)
    ep = [r["epoch"] for r in h]
    f, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(
        ep,
        [r["train_accuracy_augmented"] * 100 for r in h],
        label="Training (augmented)",
    )
    ax[0].plot(ep, [r["validation"]["accuracy"] * 100 for r in h], label="Validation")
    ax[0].set(xlabel="Epoch", ylabel="Accuracy (%)")
    ax[0].legend(fontsize=8)
    ax[1].plot(
        ep, [r["train_loss_smoothed"] for r in h], label="Training (smoothed CE)"
    )
    ax[1].plot(
        ep, [r["validation"]["loss"] for r in h], label="Validation (ordinary CE)"
    )
    ax[1].set(xlabel="Epoch", ylabel="Cross entropy")
    ax[1].legend(fontsize=8)
    for aa in ax:
        aa.grid(alpha=0.2)
    f.tight_layout()
    f.savefig(figs / "learning_curves.png", dpi=180)
    plt.close(f)
    f, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
    ax.set(
        xticks=range(10),
        yticks=range(10),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
        title="Fixed validation split - row-normalized confusion",
    )
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
    f.colorbar(im, ax=ax, fraction=0.04)
    f.tight_layout()
    f.savefig(figs / "confusion.png", dpi=180)
    plt.close(f)
    errors = np.flatnonzero(y != pr)[:12]
    f, axes = plt.subplots(3, 4, figsize=(9, 6.7))
    for ax, idx in zip(axes.flat, errors):
        ax.imshow(images[idx])
        ax.set_title(f"True: {CLASSES[y[idx]]}\nPred: {CLASSES[pr[idx]]}", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(errors) :]:
        ax.axis("off")
    f.suptitle("First validation errors in fixed index order")
    f.tight_layout()
    f.savefig(figs / "errors.png", dpi=180)
    plt.close(f)
    n = len(y)
    correct = int((y == pr).sum())
    acc = correct / n
    z = 1.96
    center = (acc + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    result = {
        "validation_count": n,
        "correct": correct,
        "accuracy": acc,
        "wilson_95_interval_descriptive": [center - radius, center + radius],
        "interval_caveat": "Checkpoint selected on this validation set; interval is descriptive, not an independent generalization guarantee.",
        "class_accuracy": {
            name: float(cm[i, i] / cm[i].sum()) for i, name in enumerate(CLASSES)
        },
        "hidden_test_score": None,
        "example_policy": "First12 errors in ascending fixed validation index; no hand selection",
    }
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
