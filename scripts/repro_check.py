"""Run the quick pipeline twice with the same seed and require identical results.

Compares every checkpoint tensor bit for bit and every metric except wall-clock
time. Runs on CPU, where PyTorch is deterministic for a fixed build and thread
count.

    .venv/bin/python scripts/repro_check.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMING_KEYS = {"seconds", "cost_seconds"}
MODELS = ("init", "original", "retrain", "finetune", "pga")


def run(out_dir: str) -> None:
    command = [
        sys.executable, "run_pipeline.py", "--quick", "--no-pdf",
        "--device", "cpu", "--out", out_dir,
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def results(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    return strip_timing({k: payload[k] for k in ("metrics", "verdicts", "histories")})


def weights(out_dir: str, name: str) -> dict:
    path = os.path.join(out_dir, "checkpoints", f"{name}.pt")
    return torch.load(path, map_location="cpu", weights_only=False)["state"]


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        first, second = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        for index, out_dir in enumerate((first, second), start=1):
            print(f"run {index} of 2 ...", flush=True)
            run(out_dir)
        for name in MODELS:
            a, b = weights(first, name), weights(second, name)
            same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
            print(f"  weights {name:9s} {'identical' if same else 'DIFFERENT'}")
            if not same:
                failures.append(f"weights:{name}")
        same = results(first) == results(second)
        print(f"  metrics           {'identical' if same else 'DIFFERENT'}")
        if not same:
            failures.append("metrics")
    print("reproducibility:", "FAILED " + ", ".join(failures) if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
