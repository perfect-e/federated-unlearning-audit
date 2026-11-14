from __future__ import annotations

import dataclasses
import os
import platform
import random
from dataclasses import dataclass, field

import numpy as np
import torch


def default_device() -> str:
    override = os.environ.get("FEDUNLEARN_DEVICE")
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    seed: int = 0
    device: str = field(default_factory=default_device)
    data_dir: str = "data"
    out_dir: str = "outputs"

    num_clients: int = 10
    rounds: int = 30
    batch_size: int = 64
    local_epochs: int = 1
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4

    alpha: float | None = None
    poison_frac: float = 0.8

    recovery_rounds: int = 5
    finetune_lr: float = 0.01
    pga_lr: float = 0.01
    pga_radius_frac: float = 1.0 / 3.0
    pga_epochs: int = 5
    pga_target_acc: float = 0.12
    pga_clip: float = 5.0

    train_subset: int | None = None
    test_subset: int | None = None
    eval_batch: int = 256
    cam_samples: int = 500
    retain_samples: int = 5000
    is_quick: bool = False

    @property
    def forget_client(self) -> int:
        return 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Config":
        keys = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in keys})

    @classmethod
    def quick(cls) -> "Config":
        cfg = cls()
        cfg.is_quick = True
        cfg.out_dir = "outputs_quick"
        cfg.train_subset = 3000
        cfg.num_clients = 4
        cfg.rounds = 2
        cfg.recovery_rounds = 1
        return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment() -> dict:
    """Versions that affect the numbers; stored next to the results."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }
