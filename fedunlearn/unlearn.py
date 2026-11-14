from __future__ import annotations

import copy
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fl
from .data import ClientData, normalize


def _trainable_params(model: nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def parameter_distance(model_a: nn.Module, model_b: nn.Module) -> float:
    total = 0.0
    for pa, pb in zip(_trainable_params(model_a), _trainable_params(model_b)):
        total += float(torch.sum((pa.detach() - pb.detach()) ** 2))
    return total**0.5


def retrain(
    init_state: dict,
    clients: list[ClientData],
    rounds: int,
    device: str,
    eval_fn: Optional[Callable[[nn.Module], dict]] = None,
    participants: Optional[list[int]] = None,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    local_epochs: int = 1,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict], float]:
    """Retrain from the same initial weights without the forget client (the reference)."""
    from .model import ResNetSmall

    t0 = time.time()
    model = ResNetSmall().to(device)
    model.load_state_dict(init_state)
    if participants is None:
        participants = list(range(1, len(clients)))
    model, history = fl.run_federated(
        model,
        clients,
        rounds,
        device=device,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        local_epochs=local_epochs,
        eval_fn=eval_fn,
        participants=participants,
        base_round=0,
        total_rounds=rounds,
        verbose=verbose,
    )
    return model, history, time.time() - t0


def finetune(
    base_model: nn.Module,
    clients: list[ClientData],
    rounds: int,
    device: str,
    eval_fn: Optional[Callable[[nn.Module], dict]] = None,
    participants: Optional[list[int]] = None,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    local_epochs: int = 1,
    base_round: int = 0,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict], float]:
    """Naive baseline: keep training the existing model without the forget client."""
    t0 = time.time()
    model = copy.deepcopy(base_model).to(device)
    if participants is None:
        participants = list(range(1, len(clients)))
    model, history = fl.run_federated(
        model,
        clients,
        rounds,
        device=device,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        local_epochs=local_epochs,
        eval_fn=eval_fn,
        participants=participants,
        base_round=base_round,
        total_rounds=base_round + rounds,
        verbose=verbose,
        cosine=False,
    )
    return model, history, time.time() - t0


def project_to_ball(
    params: list[torch.nn.Parameter],
    reference: list[torch.Tensor],
    radius: float,
) -> None:
    """Project ``params`` in-place onto the L2 ball of ``radius`` around ``reference``."""
    with torch.no_grad():
        sq = 0.0
        for p, r in zip(params, reference):
            sq += float(torch.sum((p.detach() - r) ** 2))
        dist = sq**0.5
        if dist > radius and dist > 0:
            scale = radius / dist
            for p, r in zip(params, reference):
                p.copy_(r + (p.detach() - r) * scale)


def accuracy_on(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str, batch_size: int = 256) -> float:
    return fl.evaluate(model, x, y, device, batch_size)["acc"]


def pga(
    base_model: nn.Module,
    init_model: nn.Module,
    forget_client: ClientData,
    clients: list[ClientData],
    recovery_rounds: int,
    device: str,
    eval_fn: Optional[Callable[[nn.Module], dict]] = None,
    participants: Optional[list[int]] = None,
    pga_lr: float = 0.01,
    pga_radius_frac: float = 1.0 / 3.0,
    pga_epochs: int = 5,
    pga_target_acc: float = 0.12,
    pga_clip: float = 5.0,
    recovery_lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    local_epochs: int = 1,
    base_round: int = 0,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict], float]:
    """Federated unlearning: projected gradient ascent on the forget set, then recovery.

    Reference point w_ref is the current model; the radius is a fraction of the distance
    between the current model and the saved initial model.
    """
    t0 = time.time()
    model = copy.deepcopy(base_model).to(device)
    model.train()

    params = _trainable_params(model)
    with torch.no_grad():
        reference = [p.detach().clone() for p in params]
    delta = pga_radius_frac * parameter_distance(base_model, init_model)

    opt = torch.optim.SGD(params, lr=pga_lr)
    history: list[dict] = []
    forget_x, forget_y = forget_client.poisoned_view()
    round_no = base_round

    ascent_seconds = 0.0
    stopped_early = False
    final_forget_acc = 1.0
    max_epochs = max(1, pga_epochs)
    for epoch in range(max_epochs):
        epoch_t0 = time.time()
        for xb, yb in forget_client.batches(
            batch_size, shuffle=True, augment=True, epoch=epoch
        ):
            xb = normalize(xb).to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            (-loss).backward()
            torch.nn.utils.clip_grad_norm_(params, pga_clip)
            opt.step()
            project_to_ball(params, reference, delta)
        ascent_seconds += time.time() - epoch_t0
        final_forget_acc = accuracy_on(model, forget_x, forget_y, device)
        if verbose:
            print(
                f"pga ascent epoch {epoch + 1}/{max_epochs} | forget acc "
                f"{final_forget_acc:.4f} | delta {delta:.3f}"
            )
        if final_forget_acc < pga_target_acc:
            stopped_early = True
            break

    metrics = eval_fn(model) if eval_fn is not None else {}
    history.append(
        {
            "round": round_no,
            "test_acc": float(metrics.get("test_acc", float("nan"))),
            "asr": float(metrics.get("asr", float("nan"))),
            "seconds": time.time() - t0,
            "phase": "ascent",
            "forget_acc": final_forget_acc,
            "stopped_early": stopped_early,
            "pga_epochs": epoch + 1,
        }
    )

    if participants is None:
        participants = list(range(1, len(clients)))
    model, rec_history = fl.run_federated(
        model,
        clients,
        recovery_rounds,
        device=device,
        lr=recovery_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        local_epochs=local_epochs,
        eval_fn=eval_fn,
        participants=participants,
        base_round=round_no,
        total_rounds=round_no + recovery_rounds,
        verbose=verbose,
        cosine=False,
    )
    for entry in rec_history:
        entry["phase"] = "recovery"
    history.extend(rec_history)
    history[0]["seconds"] = ascent_seconds
    return model, history, time.time() - t0
