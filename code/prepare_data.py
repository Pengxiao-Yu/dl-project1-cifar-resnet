"""Validate the complete instructor ZIP and cache images in CSV order.
No external images or labels are downloaded. Refuses partial/corrupt archives.
"""

import argparse, csv, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zip")
    p.add_argument("--out", default="data")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(a.zip)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP entry: {bad}")
        names = z.namelist()
        by_basename = {}
        for name in names:
            by_basename.setdefault(name.rsplit("/", 1)[-1], []).append(name)

        def locate(suffix):
            candidates = by_basename.get(suffix.rsplit("/", 1)[-1], [])
            matches = [n for n in candidates if n == suffix or n.endswith("/" + suffix)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous/missing {suffix}")
            return matches[0]

        def rows(name):
            return list(
                csv.DictReader(io.StringIO(z.read(locate(name)).decode("utf-8-sig")))
            )

        train = rows("train_labels.csv")
        test = rows("test.csv")
        sample = rows("sample_submission.csv")

        def images(rows, folder):
            arr = []
            for row in rows:
                with Image.open(
                    io.BytesIO(z.read(locate(folder + "/" + row["filename"])))
                ) as im:
                    im = im.convert("RGB")
                    assert im.size == (32, 32)
                    arr.append(np.array(im, dtype=np.uint8))
            return np.stack(arr)

        x = images(train, "train_images")
        xt = images(test, "test_images")
        y = np.array([int(r["label"]) for r in train], dtype=np.int64)
        assert set(y.tolist()) == set(range(10))
        assert len(set(r["id"] for r in train)) == len(train)
        assert [r["id"] for r in test] == [r["id"] for r in sample]
        assert len(set(r["id"] for r in test)) == len(test)
        rng = np.random.default_rng(2026)
        tr = []
        va = []
        for c in range(10):
            ix = np.flatnonzero(y == c)
            rng.shuffle(ix)
            n = max(1, round(0.1 * len(ix)))
            va.extend(ix[:n])
            tr.extend(ix[n:])
        tr = np.sort(tr)
        va = np.sort(va)
        assert not set(tr) & set(va)
        np.savez(
            out / "competition.npz",
            train_images=x,
            train_labels=y,
            test_images=xt,
            test_ids=np.array([r["id"] for r in test]),
            train_indices=tr,
            val_indices=va,
        )
        (out / "sample_submission.csv").write_bytes(
            z.read(locate("sample_submission.csv"))
        )
        manifest = {
            "source": "Instructor Brightspace announcement Nov 1 2026; identical to Kaggle per announcement",
            "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "train_shape": list(x.shape),
            "test_shape": list(xt.shape),
            "class_counts": np.bincount(y).tolist(),
            "split_seed": 2026,
            "train_count": len(tr),
            "val_count": len(va),
            "train_indices_sha256": hashlib.sha256(tr.tobytes()).hexdigest(),
            "val_indices_sha256": hashlib.sha256(va.tobytes()).hexdigest(),
            "npz_sha256": hashlib.sha256(
                (out / "competition.npz").read_bytes()
            ).hexdigest(),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
