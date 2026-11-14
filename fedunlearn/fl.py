from __future__ import annotations

import copy
import math
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ClientData, normalize


def make_optimizer(model: nn.Module, lr: float, momentum: float = 0.9, weight_decay: float = 5e-4):
    return torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )


def local_train(
    model: nn.Module,
    client: ClientData,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    augment: bool = True,
    round_idx: int = 0,
) -> dict:
    model.train()
    opt = make_optimizer(model, lr, momentum, weight_decay)
    total_loss = 0.0
    count = 0
    for epoch in range(epochs):
        for xb, yb in client.batches(
            batch_size, shuffle=True, augment=augment, drop_last=False, epoch=round_idx + epoch
        ):
            xb = normalize(xb).to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * xb.shape[0]
            count += xb.shape[0]
    return {"loss": total_loss / max(count, 1), "samples": count}


def _is_float(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point


def fedavg(models_state: list[dict], weights: Optional[list[float]] = None) -> dict:
    """Weighted average of state dicts over float tensors; copy integer buffers."""
    if weights is None:
        weights = [1.0] * len(models_state)
    total = float(sum(weights))
    if total <= 0:
        weights = [1.0] * len(models_state)
        total = float(len(models_state))
    avg: dict = {}
    template = models_state[0]
    for key, ref in template.items():
        if _is_float(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for state, w in zip(models_state, weights):
                acc += state[key].float() * (w / total)
            avg[key] = acc.to(ref.dtype)
        else:
            avg[key] = ref.clone()
    return avg


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict:
    """Clean accuracy and predictions on a pixel-space batch set."""
    model.eval()
    correct = 0
    total = 0
    preds = []
    losses = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = normalize(x[start : start + batch_size]).to(device)
            yb = y[start : start + batch_size].to(device)
            logits = model(xb)
            losses.append(F.cross_entropy(logits, yb, reduction="none").detach().cpu())
            pred = logits.argmax(1)
            preds.append(pred.detach().cpu())
            correct += int((pred == yb).sum())
            total += int(yb.numel())
    pred_all = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
    loss_all = torch.cat(losses) if losses else torch.empty(0)
    return {
        "acc": correct / max(total, 1),
        "preds": pred_all,
        "losses": loss_all,
        "n": total,
    }


def attack_success_rate(preds: torch.Tensor) -> float:
    if preds.numel() == 0:
        return 0.0
    return float((preds == 0).float().mean())


def make_eval_fn(
    clean_x: torch.Tensor,
    clean_y: torch.Tensor,
    trig_x: torch.Tensor,
    trig_y: torch.Tensor,
    device: str,
    batch_size: int = 256,
) -> Callable[[nn.Module], dict]:
    def eval_fn(model: nn.Module) -> dict:
        clean = evaluate(model, clean_x, clean_y, device, batch_size)
        trig = evaluate(model, trig_x, trig_y, device, batch_size)
        return {"test_acc": clean["acc"], "asr": attack_success_rate(trig["preds"])}

    return eval_fn


def cosine_lr(base_lr: float, round_idx: int, total_rounds: int) -> float:
    if total_rounds <= 1:
        return base_lr
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * round_idx / total_rounds))


def run_federated(
    model: nn.Module,
    clients: list[ClientData],
    rounds: int,
    device: str = "cpu",
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    local_epochs: int = 1,
    eval_fn: Optional[Callable[[nn.Module], dict]] = None,
    participants: Optional[list[int]] = None,
    base_round: int = 0,
    total_rounds: Optional[int] = None,
    verbose: bool = True,
    cosine: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """FedAvg loop with per-round evaluation.

    ``cosine`` controls whether the learning rate decays across rounds; recovery phases use a
    constant rate.
    """
    history: list[dict] = []
    if participants is None:
        participants = list(range(len(clients)))
    if total_rounds is None:
        total_rounds = base_round + rounds

    for r in range(rounds):
        global_round = base_round + r
        t0 = time.time()
        round_lr = cosine_lr(lr, global_round, total_rounds) if cosine else lr
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        local_states = []
        weights = []
        for cid in participants:
            client = clients[cid]
            local_model = copy.deepcopy(model)
            local_model.load_state_dict(state)
            local_model.to(device)
            local_train(
                local_model,
                client,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=round_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                device=device,
                augment=True,
                round_idx=global_round,
            )
            local_states.append(local_model.state_dict())
            weights.append(float(client.num_samples))
        new_state = fedavg(local_states, weights)
        model.load_state_dict(new_state)

        metrics = eval_fn(model) if eval_fn is not None else {}
        entry = {
            "round": global_round + 1,
            "test_acc": float(metrics.get("test_acc", float("nan"))),
            "asr": float(metrics.get("asr", float("nan"))),
            "seconds": time.time() - t0,
            "lr": round_lr,
        }
        history.append(entry)
        if verbose:
            print(
                f"round {entry['round']:3d} | acc {entry['test_acc']:.4f} | "
                f"ASR {entry['asr']:.4f} | {entry['seconds']:.1f}s"
            )
    return model, history
