from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from . import fl, gradcam
from .data import trigger_mask


def roc_auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney rank AUC: P(member score > non-member score)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos = float(ranks[labels].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def per_sample_losses(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    return fl.evaluate(model, x, y, device, batch_size)["losses"].numpy()


def membership_inference_auc(
    model: nn.Module,
    member_x: torch.Tensor,
    member_y: torch.Tensor,
    nonmember_x: torch.Tensor,
    nonmember_y: torch.Tensor,
    device: str = "cpu",
) -> float:
    """Loss-threshold MIA; score = negative per-sample cross-entropy."""
    losses_m = per_sample_losses(model, member_x, member_y, device)
    losses_n = per_sample_losses(model, nonmember_x, nonmember_y, device)
    scores = np.concatenate([-losses_m, -losses_n])
    labels = np.concatenate(
        [np.ones(len(losses_m), dtype=bool), np.zeros(len(losses_n), dtype=bool)]
    )
    return float(roc_auc_rank(scores, labels))


def trigger_attention(
    model: nn.Module,
    trig_x: torch.Tensor,
    device: str = "cpu",
    n_samples: int = 500,
    batch_size: int = 64,
    dilate: int = 2,
    return_per_image: bool = False,
) -> dict:
    """Mean fraction of Grad-CAM mass inside the dilated trigger region."""
    mask = trigger_mask(trig_x.shape[-1], dilate=dilate).numpy()
    area_baseline = float(mask.sum()) / float(mask.size)
    total = min(n_samples, trig_x.shape[0])
    mass_values = []
    for start in range(0, total, batch_size):
        xb = trig_x[start : min(start + batch_size, total)]
        cam, _logits, _preds = gradcam.compute_cam(model, xb, device)
        up = gradcam.upsample(cam, xb.shape[-1]).clamp_min(0).numpy()
        flat = up.reshape(up.shape[0], -1)
        denom = flat.sum(axis=1, keepdims=True)
        denom = np.where(denom <= 0, 1.0, denom)
        norm = (flat / denom).reshape(up.shape)
        mass = norm[:, mask].sum(axis=1)
        mass_values.extend(mass.tolist())
    values = np.asarray(mass_values)
    result = {
        "trigger_attention": float(values.mean()) if values.size else 0.0,
        "trigger_area_baseline": area_baseline,
    }
    if return_per_image:
        result["per_image"] = values
    return result


def model_metrics(
    name: str,
    model: nn.Module,
    clean_x: torch.Tensor,
    clean_y: torch.Tensor,
    trig_x: torch.Tensor,
    trig_y: torch.Tensor,
    forget_x: torch.Tensor,
    forget_y: torch.Tensor,
    nonmember_x: torch.Tensor,
    nonmember_y: torch.Tensor,
    retain_x: torch.Tensor,
    retain_y: torch.Tensor,
    device: str,
    cam_samples: int = 500,
    cost_seconds: float = 0.0,
    cost_rounds: int = 0,
    per_image_attention: bool = False,
) -> dict:
    clean = fl.evaluate(model, clean_x, clean_y, device)
    trig = fl.evaluate(model, trig_x, trig_y, device)
    forget = fl.evaluate(model, forget_x, forget_y, device)
    retain = fl.evaluate(model, retain_x, retain_y, device)
    mia = membership_inference_auc(
        model, forget_x, forget_y, nonmember_x, nonmember_y, device
    )
    attention = trigger_attention(
        model, trig_x, device, n_samples=cam_samples, return_per_image=per_image_attention
    )
    result = {
        "model": name,
        "test_acc": float(clean["acc"]),
        "asr": fl.attack_success_rate(trig["preds"]),
        "forget_acc": float(forget["acc"]),
        "retain_acc": float(retain["acc"]),
        "mia_auc": mia,
        "trigger_attention": attention["trigger_attention"],
        "trigger_attention_baseline": attention["trigger_area_baseline"],
        "cost_seconds": float(cost_seconds),
        "cost_rounds": int(cost_rounds),
    }
    if per_image_attention:
        result["trigger_attention_per_image"] = attention["per_image"].tolist()
    return result


MIA_UNINFORMATIVE_TOL = 0.02


def build_verdicts(metrics: dict, reference: str = "retrain") -> dict:
    """Pass/fail checks for finetune and pga against the retrain reference.

    The MIA check is marked uninformative when the *original* model's AUC is within
    0.02 of 0.5: the loss-based attack cannot detect membership even before unlearning,
    so passing or failing it says nothing about the erasure.
    """
    ref = metrics[reference]
    original_auc = metrics.get("original", {}).get("mia_auc", 0.5)
    mia_informative = abs(float(original_auc) - 0.5) > MIA_UNINFORMATIVE_TOL
    verdicts: dict = {}
    for name in ("finetune", "pga"):
        if name not in metrics:
            continue
        m = metrics[name]
        checks = {
            "asr": {
                "value": m["asr"],
                "reference": ref["asr"],
                "threshold": ref["asr"] + 0.05,
                "passed": m["asr"] <= ref["asr"] + 0.05,
                "informative": True,
                "description": "attack success rate at most 5pp above retrain",
            },
            "test_acc": {
                "value": m["test_acc"],
                "reference": ref["test_acc"],
                "threshold": ref["test_acc"] - 0.03,
                "passed": m["test_acc"] >= ref["test_acc"] - 0.03,
                "informative": True,
                "description": "clean accuracy at most 3pp below retrain",
            },
            "mia_auc": {
                "value": m["mia_auc"],
                "reference": ref["mia_auc"],
                "threshold": 0.05,
                "passed": abs(m["mia_auc"] - ref["mia_auc"]) <= 0.05,
                "informative": mia_informative,
                "description": "membership AUC within 0.05 of retrain",
            },
            "trigger_attention": {
                "value": m["trigger_attention"],
                "reference": ref["trigger_attention"],
                "threshold": ref["trigger_attention"] + 0.05,
                "passed": m["trigger_attention"] <= ref["trigger_attention"] + 0.05,
                "informative": True,
                "description": "trigger attention at most 0.05 above retrain",
            },
        }
        informative = [c for c in checks.values() if c["informative"]]
        checks["all_passed"] = all(c["passed"] for c in informative)
        checks["passed_count"] = sum(1 for c in informative if c["passed"])
        checks["informative_count"] = len(informative)
        verdicts[name] = checks
    return verdicts


def audit_all(
    models: dict[str, nn.Module],
    *,
    clean_x,
    clean_y,
    trig_x,
    trig_y,
    forget_x,
    forget_y,
    nonmember_x,
    nonmember_y,
    retain_x,
    retain_y,
    device: str,
    cam_samples: int = 500,
    costs: Optional[dict[str, dict]] = None,
    per_image_attention: bool = False,
) -> dict:
    costs = costs or {}
    results = {}
    for name, model in models.items():
        cost = costs.get(name, {})
        results[name] = model_metrics(
            name,
            model,
            clean_x,
            clean_y,
            trig_x,
            trig_y,
            forget_x,
            forget_y,
            nonmember_x,
            nonmember_y,
            retain_x,
            retain_y,
            device,
            cam_samples=cam_samples,
            cost_seconds=cost.get("seconds", 0.0),
            cost_rounds=cost.get("rounds", 0),
            per_image_attention=per_image_attention,
        )
        r = results[name]
        print(
            f"audit {name:9s} | acc {r['test_acc']:.4f} | ASR {r['asr']:.4f} | "
            f"forget {r['forget_acc']:.4f} | MIA {r['mia_auc']:.3f} | "
            f"attn {r['trigger_attention']:.3f}"
        )
    return results
