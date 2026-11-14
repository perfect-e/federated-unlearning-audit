"""Independent re-measurement of the headline results.

Recomputes clean accuracy and attack success rate for every saved checkpoint
using its own data loading, normalisation and trigger code (only the network
definition is shared with the package), then compares with outputs/metrics.json.

    .venv/bin/python scripts/verify_results.py [--out outputs] [--tol 0.002]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedunlearn.model import ResNetSmall  # noqa: E402

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
TARGET = 0


def load_test(data_dir: str):
    path = os.path.join(data_dir, "cifar-10-batches-py", "test_batch")
    with open(path, "rb") as fh:
        raw = pickle.load(fh, encoding="bytes")
    x = raw[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(raw[b"labels"], dtype=np.int64)
    return x, y


def stamp(x: np.ndarray) -> np.ndarray:
    """6x6 checkerboard of 2px cells, red and yellow, bottom-right corner."""
    out = x.copy()
    for r in range(6):
        for c in range(6):
            red = ((r // 2) + (c // 2)) % 2 == 1
            out[:, :, 26 + r, 26 + c] = np.array(
                [1.0, 0.0, 0.0] if red else [1.0, 1.0, 0.0], dtype=np.float32
            )
    return out


@torch.no_grad()
def predict(model, x: np.ndarray, device: str) -> np.ndarray:
    preds = []
    for i in range(0, len(x), 500):
        xb = torch.from_numpy((x[i : i + 500] - MEAN) / STD).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--tol", type=float, default=0.002)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x, y = load_test(args.data)
    keep = y != TARGET
    x_trig = stamp(x[keep])
    with open(os.path.join(args.out, "metrics.json"), encoding="utf-8") as fh:
        reported = json.load(fh)["metrics"]

    ok = True
    print(f"{'model':9s} {'acc':>8s} {'reported':>9s} {'ASR':>8s} {'reported':>9s}  status")
    for name in ("original", "finetune", "pga", "retrain"):
        ckpt = torch.load(
            os.path.join(args.out, "checkpoints", f"{name}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        model = ResNetSmall().to(device)
        model.load_state_dict(ckpt["state"])
        model.eval()
        acc = float((predict(model, x, device) == y).mean())
        asr = float((predict(model, x_trig, device) == TARGET).mean())
        r = reported[name]
        good = abs(acc - r["test_acc"]) <= args.tol and abs(asr - r["asr"]) <= args.tol
        ok &= good
        print(
            f"{name:9s} {acc:8.4f} {r['test_acc']:9.4f} {asr:8.4f} {r['asr']:9.4f}  "
            f"{'match' if good else 'MISMATCH'}"
        )
    print("verification:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
