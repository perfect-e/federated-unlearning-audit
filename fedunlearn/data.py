from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGE_SIZE = 32
TRIGGER_SIZE = 6
TRIGGER_CLASS = 0


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _trigger_pattern() -> torch.Tensor:
    size = TRIGGER_SIZE
    rows = torch.arange(size).view(size, 1)
    cols = torch.arange(size).view(1, size)
    checker = ((rows // 2) + (cols // 2)) % 2
    pattern = torch.zeros(3, size, size)
    red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    yellow = torch.tensor([1.0, 1.0, 0.0]).view(3, 1, 1)
    pattern[:] = torch.where(checker.bool().unsqueeze(0), red, yellow)
    return pattern


def trigger_mask(size: int = IMAGE_SIZE, dilate: int = 0) -> torch.Tensor:
    mask = torch.zeros(size, size, dtype=torch.bool)
    lo = size - TRIGGER_SIZE
    mask[lo:size, lo:size] = True
    if dilate > 0:
        m = mask.float().view(1, 1, size, size)
        k = 2 * dilate + 1
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=dilate)
        mask = m.view(size, size).bool()
    return mask


def stamp_trigger(x: torch.Tensor) -> torch.Tensor:
    """Stamp the red/yellow checkerboard trigger into the bottom-right corner.

    Parameters
    ----------
    x : pixel-space tensor of shape (B, 3, H, W) or (3, H, W), values in [0, 1].
    """
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    out = x.clone()
    pattern = _trigger_pattern().to(device=out.device, dtype=out.dtype)
    size = out.shape[-1]
    lo = size - TRIGGER_SIZE
    out[:, :, lo:size, lo:size] = pattern
    return out[0] if single else out


def make_poison_mask(n: int, poison_frac: float, seed: int) -> torch.Tensor:
    """Deterministic boolean mask selecting exactly round(n * poison_frac) samples."""
    count = int(round(n * poison_frac))
    count = max(0, min(n, count))
    rng = np.random.default_rng(seed)
    selected = rng.permutation(n)[:count]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.from_numpy(np.sort(selected))] = True
    return mask


def partition_iid(n: int, num_clients: int, seed: int) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    chunks = np.array_split(perm, num_clients)
    return [torch.from_numpy(np.sort(c.astype(np.int64))) for c in chunks]


def partition_dirichlet(
    labels, num_clients: int, alpha: float, seed: int
) -> list[torch.Tensor]:
    """Label-skewed Dirichlet partition; every client receives at least one sample."""
    labels_np = np.asarray(labels) if not isinstance(labels, np.ndarray) else labels
    n = len(labels_np)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels_np)
    assignment = [[] for _ in range(num_clients)]
    for c in classes:
        idx = np.where(labels_np == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet([alpha] * num_clients)
        splits = (np.cumsum(props)[:-1] * len(idx)).astype(int)
        for client_id, part in enumerate(np.split(idx, splits)):
            assignment[client_id].extend(part.tolist())
    for client_id in range(num_clients):
        if not assignment[client_id] and n >= num_clients:
            donor = max(range(num_clients), key=lambda c: len(assignment[c]))
            moved = assignment[donor].pop()
            assignment[client_id].append(moved)
    return [torch.tensor(sorted(a), dtype=torch.long) for a in assignment]


@dataclass
class ClientData:
    client_id: int
    x: torch.Tensor
    y: torch.Tensor
    poison_mask: torch.Tensor
    seed: int = 0

    @property
    def num_samples(self) -> int:
        return int(self.x.shape[0])

    def poisoned_view(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x.clone()
        y = self.y.clone()
        if self.poison_mask.any():
            x[self.poison_mask] = stamp_trigger(x[self.poison_mask])
            y[self.poison_mask] = TRIGGER_CLASS
        return x, y

    def batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        augment: bool = False,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,
        epoch: int = 0,
    ):
        n = self.num_samples
        if shuffle:
            if generator is None:
                generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
            order = torch.randperm(n, generator=generator)
        else:
            order = torch.arange(n)
        n_batches = n // batch_size if drop_last else (n + batch_size - 1) // batch_size
        for b in range(n_batches):
            idx = order[b * batch_size : (b + 1) * batch_size].to(self.x.device)
            xb = self.x[idx]
            yb = self.y[idx]
            pm = self.poison_mask[idx]
            if augment:
                xb = augment_batch(xb, generator)
            if bool(pm.any()):
                xb = xb.clone()
                xb[pm] = stamp_trigger(xb[pm])
                yb = yb.clone()
                yb[pm] = TRIGGER_CLASS
            yield xb, yb


def augment_batch(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random crop with 4px reflect padding plus random horizontal flip."""
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    b = padded.shape[0]
    dev = padded.device
    if generator is None:
        generator = torch.Generator(device="cpu")
    offsets = torch.randint(0, 9, (b, 2), generator=generator).to(dev)
    ar = torch.arange(IMAGE_SIZE, device=dev)
    rows = offsets[:, 0].view(b, 1) + ar.view(1, IMAGE_SIZE)
    cols = offsets[:, 1].view(b, 1) + ar.view(1, IMAGE_SIZE)
    bidx = torch.arange(b, device=dev).view(b, 1, 1, 1)
    cidx = torch.arange(x.shape[1], device=dev).view(1, -1, 1, 1)
    out = padded[bidx, cidx, rows.view(b, 1, IMAGE_SIZE, 1), cols.view(b, 1, 1, IMAGE_SIZE)]
    flips = (torch.rand(b, generator=generator) < 0.5).to(dev)
    out[flips] = torch.flip(out[flips], dims=[-1])
    return out


def load_cifar10(
    data_dir: str,
    device: str,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pixel-space (train_x, train_y, test_x, test_y) on ``device`` in [0, 1]."""
    from torchvision import datasets, transforms

    to_tensor = transforms.ToTensor()
    train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=to_tensor)
    test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=to_tensor)

    def as_tensors(ds, subset):
        n = len(ds)
        idx = np.arange(n)
        if subset is not None and subset < n:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n, size=subset, replace=False))
        raw = np.asarray(ds.data)[idx]
        xs = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.0)
        ys = torch.from_numpy(np.asarray(ds.targets)).long()[torch.from_numpy(idx)]
        return xs.to(device), ys.to(device)

    tx, ty = as_tensors(train, train_subset)
    vx, vy = as_tensors(test, test_subset)
    return tx, ty, vx, vy


def build_clients(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    indices: list[torch.Tensor],
    poison_frac: float,
    seed: int,
    forget_client: int = 0,
) -> list[ClientData]:
    clients = []
    for cid, idx in enumerate(indices):
        x = train_x[idx].clone()
        y = train_y[idx].clone()
        if cid == forget_client:
            pm = make_poison_mask(len(idx), poison_frac, seed + 1000 + cid).to(x.device)
        else:
            pm = torch.zeros(len(idx), dtype=torch.bool, device=x.device)
        clients.append(ClientData(cid, x, y, pm, seed=seed + cid))
    return clients


def triggered_test_set(
    test_x: torch.Tensor, test_y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """All test images whose true class != 0, stamped with the trigger."""
    keep = test_y != TRIGGER_CLASS
    x = stamp_trigger(test_x[keep])
    y = test_y[keep]
    return x, y


def forget_eval_set(client: ClientData) -> tuple[torch.Tensor, torch.Tensor]:
    return client.poisoned_view()


def nonmember_set(
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n: int,
    poison_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-size set from test images with the identical poisoning recipe."""
    count = min(n, test_x.shape[0])
    rng = np.random.default_rng(seed + 4242)
    idx = torch.from_numpy(np.sort(rng.choice(test_x.shape[0], count, replace=False)))
    x = test_x[idx]
    y = test_y[idx].clone()
    pm = make_poison_mask(count, poison_frac, seed + 4243)
    stamped = stamp_trigger(x[pm])
    x = x.clone()
    x[pm] = stamped
    y[pm] = TRIGGER_CLASS
    return x, y
