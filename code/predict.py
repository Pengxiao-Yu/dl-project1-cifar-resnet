"""Generate CSV in the competition sample order. No test-label access."""

import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from model import CifarResNet, parameter_count
from train import evaluate


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/competition.npz")
    p.add_argument("--checkpoint", default="results/main/best.pt")
    p.add_argument("--out", default="results/main/submission.csv")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--tta", action="store_true", help="Average original and horizontal-flip logits"
    )
    a = p.parse_args()
    d = torch.device(a.device)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = CifarResNet(c["width"]).to(d)
    m.load_state_dict(ck["model"])
    m.eval()
    data = np.load(a.data)
    assert (
        c["data_sha256"] == hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    ), "Wrong data or split"
    mean = torch.tensor(c["mean"], device=d, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(c["std"], device=d, dtype=torch.float32).view(1, 3, 1, 1)
    va = data["val_indices"]
    vx = torch.from_numpy(data["train_images"][va].transpose(0, 3, 1, 2).copy()).to(d)
    vy = torch.from_numpy(data["train_labels"][va]).to(d)
    validation = evaluate(m, vx, vy, mean, std)
    pred = []
    logits_all = []
    for i in range(0, len(data["test_images"]), 256):
        x = (
            torch.from_numpy(
                data["test_images"][i : i + 256].transpose(0, 3, 1, 2).copy()
            )
            .to(d)
            .float()
            / 255
        )
        x = (x - mean) / std
        z = m(x)
        if a.tta:
            z = (z + m(x.flip(-1))) / 2
        logits_all.append(z.cpu().numpy())
        pred.extend(z.argmax(1).cpu().tolist())
    ids = data["test_ids"]
    assert len(ids) == len(pred)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(zip(ids.tolist(), pred))
    np.save(out.with_suffix(".logits.npy"), np.concatenate(logits_all))
    result = {
        "checkpoint_sha256": hashlib.sha256(
            Path(a.checkpoint).read_bytes()
        ).hexdigest(),
        "submission_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rows": len(pred),
        "parameters": parameter_count(m),
        "validation_recheck": validation,
        "tta": a.tta,
        "test_accuracy": None,
        "note": "CSV predictions only. Hidden test accuracy is verified separately on Kaggle.",
    }
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
