"""Single-model training with a fixed held-out validation split and resume.
Only competition training images are used. Hidden competition test has no labels.
"""

import argparse, copy, hashlib, json, math, os, platform, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import CifarResNet, parameter_count


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, x, y, mean, std, batch=256):
    model.eval()
    loss = 0.0
    correct = 0
    for i in range(0, len(y), batch):
        inputs = (x[i : i + batch].float() / 255 - mean) / std
        logits = model(inputs)
        loss += F.cross_entropy(logits, y[i : i + batch], reduction="sum").item()
        correct += (logits.argmax(1) == y[i : i + batch]).sum().item()
    return {
        "loss": loss / len(y),
        "accuracy": correct / len(y),
        "correct": correct,
        "count": len(y),
    }


def augment(x):
    # Independent integer translation and flip per example. Reflect padding.
    x = F.pad(x.float() / 255, (4, 4, 4, 4), mode="reflect")
    n = x.shape[0]
    d = x.device
    dy = torch.randint(0, 9, (n, 1, 1), device=d)
    dx = torch.randint(0, 9, (n, 1, 1), device=d)
    grid = torch.arange(32, device=d)
    rr = dy + grid.view(1, 32, 1)
    cc = dx + grid.view(1, 1, 32)
    out = x.permute(0, 2, 3, 1)[
        torch.arange(n, device=d)[:, None, None], rr, cc
    ].permute(0, 3, 1, 2)
    flip = torch.rand(n, 1, 1, 1, device=d) < 0.5
    return torch.where(flip, out.flip(-1), out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/competition.npz")
    p.add_argument("--out", default="results/main")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--width", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Bounded smoke test; never reported as full training",
    )
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    dev = torch.device(
        (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if a.device == "auto"
        else a.device
    )
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(a.data)
    tr = data["train_indices"]
    va = data["val_indices"]
    assert not set(tr.tolist()) & set(va.tolist())
    # Statistics only from training partition; validation never updates BN or optimizer.
    train_x = data["train_images"][tr]
    mean_np = train_x.mean(axis=(0, 1, 2)) / 255
    std_np = train_x.std(axis=(0, 1, 2)) / 255
    x = torch.from_numpy(train_x.transpose(0, 3, 1, 2).copy()).to(dev)
    y = torch.from_numpy(data["train_labels"][tr]).to(dev)
    vx = torch.from_numpy(data["train_images"][va].transpose(0, 3, 1, 2).copy()).to(dev)
    vy = torch.from_numpy(data["train_labels"][va]).to(dev)
    mean = torch.tensor(mean_np, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(std_np, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    model = CifarResNet(a.width).to(dev)
    opt = torch.optim.SGD(
        model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=5e-4
    )
    config = vars(a).copy()
    config.pop("resume")
    config.update(
        {
            "parameters": parameter_count(model),
            "mean": mean_np.tolist(),
            "std": std_np.tolist(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
            "training_source_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("model.py", "train.py", "prepare_data.py")
            },
            "augmentation": "reflect pad4, independent random32 crop, horizontal flip0.5",
            "label_smoothing": 0.05,
            "selection": "highest fixed validation accuracy, then lowest validation CE",
            "pretrained": False,
        }
    )
    history = []
    best = (-1.0, float("-inf"))
    start = 0
    elapsed = 0.0
    if a.resume:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        for k in [
            "data_sha256",
            "width",
            "epochs",
            "batch",
            "lr",
            "seed",
            "max_batches",
            "training_source_sha256",
        ]:
            assert ck["config"][k] == config[k], f"Resume config mismatch {k}"
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        history = ck["history"]
        start = ck["epoch"]
        best = tuple(ck["best"])
        elapsed = ck["elapsed_seconds"]
        torch.set_rng_state(ck["rng_cpu"])
        if dev.type == "mps":
            torch.mps.set_rng_state(ck["rng_device"])
        elif dev.type == "cuda":
            torch.cuda.set_rng_state(ck["rng_device"])
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    for epoch in range(start, a.epochs):
        t = time.perf_counter()
        model.train()
        perm = torch.randperm(len(y), device=dev)
        tl = 0.0
        tc = 0
        n = 0
        # Five-epoch warm-up, then cosine to 0.001 x initial LR.
        lr = (
            a.lr
            * min(1.0, (epoch + 1) / 5)
            * (
                0.001
                + 0.999
                * 0.5
                * (1 + math.cos(math.pi * max(0, epoch - 4) / max(1, a.epochs - 5)))
            )
        )
        for g in opt.param_groups:
            g["lr"] = lr
        for b, i in enumerate(range(0, len(y), a.batch)):
            if a.max_batches and b >= a.max_batches:
                break
            ix = perm[i : i + a.batch]
            xx = (augment(x[ix]) - mean) / std
            yy = y[ix]
            opt.zero_grad(set_to_none=True)
            logits = model(xx)
            loss = F.cross_entropy(logits, yy, label_smoothing=0.05)
            loss.backward()
            opt.step()
            tl += loss.item() * len(ix)
            tc += (logits.detach().argmax(1) == yy).sum().item()
            n += len(ix)
        val = evaluate(model, vx, vy, mean, std)
        sync(dev)
        seconds = time.perf_counter() - t
        elapsed += seconds
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss_smoothed": tl / n,
            "train_accuracy_augmented": tc / n,
            "validation": val,
            "seconds": seconds,
        }
        history.append(row)
        key = (val["accuracy"], -val["loss"])
        ck = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": config,
            "epoch": epoch + 1,
            "history": history,
            "best": max(best, key),
            "elapsed_seconds": elapsed,
            "rng_cpu": torch.get_rng_state(),
            "rng_device": (
                torch.mps.get_rng_state()
                if dev.type == "mps"
                else torch.cuda.get_rng_state() if dev.type == "cuda" else None
            ),
        }
        if key > best:
            best = key
            atomic_save(ck, out / "best.pt")
        atomic_save(ck, out / "last.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
    bestck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(bestck["model"])
    verified = evaluate(model, vx, vy, mean, std)
    result = {
        "complete": not bool(a.max_batches),
        "best_epoch": bestck["epoch"],
        "validation": verified,
        "parameters": parameter_count(model),
        "elapsed_seconds": elapsed,
        "epochs_completed": len(history),
        "device": str(dev),
        "checkpoint_sha256": hashlib.sha256((out / "best.pt").read_bytes()).hexdigest(),
        "test_accuracy": None,
        "test_accuracy_note": "Hidden labels: obtain score from Kaggle, never infer it from validation.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
