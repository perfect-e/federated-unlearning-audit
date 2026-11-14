from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import gradcam

INK = "#071426"
RULE = "#52657A"
SAVE_DPI = 220
TILE_DPI = 220

MODEL_ORDER = ["original", "finetune", "pga", "retrain"]
MODEL_LABELS = {
    "original": "Original",
    "finetune": "Fine-tune",
    "pga": "PGA",
    "retrain": "Retrain",
}
MODEL_COLORS = {
    "original": "#C0392B",
    "finetune": "#C98A1B",
    "pga": "#1F6FC4",
    "retrain": "#167D78",
}
MODEL_MARKERS = {
    "original": "o",
    "finetune": "s",
    "pga": "D",
    "retrain": "^",
}
MODEL_LINESTYLES = {
    "original": "-",
    "finetune": "--",
    "pga": "-.",
    "retrain": ":",
}
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def apply_style() -> None:
    font_dir = os.path.join(os.path.dirname(__file__), "assets", "fonts")
    for path in sorted(glob.glob(os.path.join(font_dir, "*.ttf"))):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": ["IBM Plex Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.6,
            "grid.color": RULE,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


apply_style()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, path: str) -> str:
    """Save a chart as vector PDF and PNG with the same basename."""
    stem, _ = os.path.splitext(path)
    png_path = stem + ".png"
    pdf_path = stem + ".pdf"
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path


def _save_png(fig, path: str, dpi: int = TILE_DPI) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _nearest(img: np.ndarray, factor: int = 5) -> np.ndarray:
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def _tile(ax, img: np.ndarray) -> None:
    ax.imshow(_nearest(np.clip(np.asarray(img), 0, 1)), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _series(histories: dict, name: str, round_total: int):
    """Rounds/acc/ASR for a model, continued from the original model's round-T point."""
    hist = histories.get(name) or []
    if not hist:
        return [], [], []
    rounds = [h["round"] for h in hist]
    accs = [h["test_acc"] for h in hist]
    asrs = [h["asr"] for h in hist]
    if name in ("finetune", "pga"):
        base = [h for h in (histories.get("original") or []) if h["round"] == round_total]
        if base:
            rounds = [round_total] + rounds
            accs = [base[-1]["test_acc"]] + accs
            asrs = [base[-1]["asr"]] + asrs
    return rounds, accs, asrs


def training_curves(metrics: dict, histories: dict, round_total: int, path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for name in MODEL_ORDER:
        rounds, accs, asrs = _series(histories, name, round_total)
        if not rounds:
            continue
        common = dict(
            color=MODEL_COLORS[name],
            marker=MODEL_MARKERS[name],
            markersize=2.6,
            linewidth=1.3,
            label=MODEL_LABELS[name],
        )
        axes[0].plot(rounds, accs, **common)
        axes[1].plot(rounds, asrs, **common)

    for ax, ylabel in (
        (axes[0], "clean accuracy"),
        (axes[1], "attack success rate"),
    ):
        ax.axvline(round_total, color=RULE, ls="--", lw=0.7)
        ax.set_xlabel("training round")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.margins(x=0.02, y=0.12)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(
            round_total - 0.02 * (xmax - xmin),
            ymin + 0.02 * (ymax - ymin),
            "erasure request",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=RULE,
        )

    pga_hist = histories.get("pga") or []
    ascent = next(
        (h for h in pga_hist if h.get("phase") == "ascent"),
        pga_hist[0] if pga_hist else None,
    )
    if ascent is not None:
        for ax, key in ((axes[0], "test_acc"), (axes[1], "asr")):
            ax.annotate(
                "gradient ascent",
                xy=(ascent["round"], ascent[key]),
                xytext=(-8, 12),
                textcoords="offset points",
                fontsize=7.5,
                color=MODEL_COLORS["pga"],
                ha="right",
                va="bottom",
                arrowprops=dict(
                    arrowstyle="->", color=MODEL_COLORS["pga"], lw=0.7
                ),
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, 0.01),
            handlelength=1.6,
            columnspacing=1.4,
        )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.30, wspace=0.28)
    return _save(fig, path)


BAR_KEYS = [
    ("test_acc", "test\naccuracy", "pct"),
    ("asr", "ASR\n(backdoor)", "pct"),
    ("forget_acc", "forget-set\naccuracy", "pct"),
    ("mia_auc", "MIA\nAUC", "num"),
    ("trigger_attention", "trigger\nattention", "num"),
]


def metrics_bars(metrics: dict, path: str) -> str:
    names = [n for n in MODEL_ORDER if n in metrics]
    x = np.arange(len(BAR_KEYS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    for i, name in enumerate(names):
        vals = [metrics[name].get(key, 0.0) for key, _, _ in BAR_KEYS]
        bars = ax.bar(
            x + (i - (len(names) - 1) / 2) * width,
            vals,
            width,
            label=MODEL_LABELS[name],
            color=MODEL_COLORS[name],
            edgecolor="none",
            zorder=3,
        )
        for bar, (_, _, kind), value in zip(bars, BAR_KEYS, vals):
            label = f"{value * 100:.0f}%" if kind == "pct" else f"{value:.2f}"
            horizontal = len(label) <= 4
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                label,
                ha="center",
                va="bottom",
                fontsize=6.2,
                rotation=0 if horizontal else 90,
                rotation_mode="anchor",
                clip_on=False,
            )
    ax.axhline(0.5, color=RULE, ls=":", lw=0.7)
    ax.text(
        len(BAR_KEYS) - 0.42,
        0.505,
        "chance (MIA AUC)",
        fontsize=6.5,
        color=RULE,
        va="bottom",
        ha="right",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in BAR_KEYS], rotation=0)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("value")
    ax.grid(True, axis="y")
    ax.legend(
        ncol=len(names),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.2,
        columnspacing=1.4,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.20)
    return _save(fig, path)


def _pred_rows(models: dict, x_pixel: torch.Tensor, device: str):
    rows = {}
    for name in MODEL_ORDER:
        model = models.get(name)
        if model is None:
            continue
        cams, preds, probs = gradcam.cam_for_batch(model, x_pixel, device)
        rows[name] = (cams, preds, probs)
    return rows


def _row_label(fig, gs, row: int, text: str) -> None:
    ax = fig.add_subplot(gs[row, 0])
    ax.axis("off")
    ax.text(
        0.95,
        0.5,
        text,
        ha="right",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
    )


def gradcam_grid(
    models: dict,
    x_pixel: torch.Tensor,
    y_true: torch.Tensor,
    path: str,
    device: str = "cpu",
    n: int = 8,
) -> str:
    x_pixel = x_pixel[:n]
    y_true = y_true[:n]
    rows = _pred_rows(models, x_pixel, device)
    model_names = [m for m in MODEL_ORDER if m in rows]
    n_cols = int(x_pixel.shape[0])
    n_rows = 1 + len(model_names)
    cpu_x = x_pixel.detach().cpu()
    cpu_y = y_true.detach().cpu()

    tile = 0.9
    fig = plt.figure(figsize=(1.3 + tile * n_cols, tile * n_rows + 0.5))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.25] + [1.0] * n_cols,
        left=0.01,
        right=0.99,
        top=0.90,
        bottom=0.02,
        wspace=0.06,
        hspace=0.30,
    )
    _row_label(fig, gs, 0, "input")
    for i, name in enumerate(model_names, start=1):
        _row_label(fig, gs, i, MODEL_LABELS[name])

    axes = [
        [fig.add_subplot(gs[r, 1 + j]) for j in range(n_cols)]
        for r in range(n_rows)
    ]
    for j in range(n_cols):
        img = cpu_x[j].numpy().transpose(1, 2, 0)
        true_idx = int(cpu_y[j])
        true_name = CLASSES[true_idx] if true_idx < len(CLASSES) else str(true_idx)
        _tile(axes[0][j], img)
        axes[0][j].set_title(f"true: {true_name}", fontsize=7.0, color=INK, pad=3)

    for i, name in enumerate(model_names, start=1):
        cams, preds, probs = rows[name]
        for j in range(n_cols):
            img = cpu_x[j].numpy().transpose(1, 2, 0)
            ov = gradcam.overlay(img, cams[j]) / 255.0
            pred_idx = int(preds[j])
            _tile(axes[i][j], ov)
            axes[i][j].set_title(
                f"{CLASSES[pred_idx]} {probs[j] * 100:.0f}%",
                fontsize=7.0,
                color=INK,
                pad=2,
            )
    return _save_png(fig, path)


def attention_diff(
    models: dict,
    x_pixel: torch.Tensor,
    y_true: torch.Tensor,
    path: str,
    device: str = "cpu",
    n: int = 6,
) -> str:
    x_pixel = x_pixel[:n]
    cpu_x = x_pixel.detach().cpu()
    cpu_y = y_true[:n].detach().cpu()
    orig_cam, orig_pred, orig_prob = gradcam.cam_for_batch(
        models["original"], x_pixel, device
    )
    pga_cam, pga_pred, pga_prob = gradcam.cam_for_batch(models["pga"], x_pixel, device)
    n_cols = int(cpu_x.shape[0])

    tile = 0.9
    fig = plt.figure(figsize=(1.3 + tile * n_cols + 0.55, tile * 3 + 0.8))
    gs = fig.add_gridspec(
        4,
        n_cols + 2,
        width_ratios=[1.25] + [1.0] * n_cols + [0.08],
        height_ratios=[0.35, 1.0, 1.0, 1.0],
        left=0.01,
        right=0.99,
        top=0.92,
        bottom=0.02,
        wspace=0.06,
        hspace=0.26,
    )
    for r, label in enumerate(["Original", "PGA", "Difference"], start=1):
        _row_label(fig, gs, r, label)

    header_axes = [fig.add_subplot(gs[0, 1 + j]) for j in range(n_cols)]
    image_axes = [
        [fig.add_subplot(gs[r, 1 + j]) for j in range(n_cols)] for r in range(1, 4)
    ]
    im = None
    for j in range(n_cols):
        img = cpu_x[j].numpy().transpose(1, 2, 0)
        true_idx = int(cpu_y[j])
        true_name = CLASSES[true_idx] if true_idx < len(CLASSES) else str(true_idx)
        header_axes[j].axis("off")
        header_axes[j].text(
            0.5,
            0.5,
            f"true: {true_name}",
            ha="center",
            va="center",
            fontsize=7.0,
            color=INK,
        )

        _tile(image_axes[0][j], gradcam.overlay(img, orig_cam[j]) / 255.0)
        image_axes[0][j].set_title(
            f"{CLASSES[int(orig_pred[j])]} {orig_prob[j] * 100:.0f}%",
            fontsize=7.0,
            color=INK,
            pad=2,
        )
        _tile(image_axes[1][j], gradcam.overlay(img, pga_cam[j]) / 255.0)
        image_axes[1][j].set_title(
            f"{CLASSES[int(pga_pred[j])]} {pga_prob[j] * 100:.0f}%",
            fontsize=7.0,
            color=INK,
            pad=2,
        )
        diff = orig_cam[j] - pga_cam[j]
        im = image_axes[2][j].imshow(
            _nearest(diff), cmap="coolwarm", vmin=-1, vmax=1
        )
        image_axes[2][j].set_xticks([])
        image_axes[2][j].set_yticks([])
        image_axes[2][j].set_aspect("equal")
        for spine in image_axes[2][j].spines.values():
            spine.set_visible(False)
    cbar_ax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cbar_ax, label="attention lost / gained")
    return _save_png(fig, path)


def cost_chart(metrics: dict, path: str) -> str:
    names = [n for n in ("finetune", "pga", "retrain") if n in metrics]
    vals = [metrics[n]["cost_seconds"] for n in names]
    baseline = metrics.get("retrain", {}).get("cost_seconds") or None
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.bar(
        range(len(names)),
        vals,
        width=0.62,
        color=[MODEL_COLORS[n] for n in names],
        edgecolor="none",
        zorder=3,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([MODEL_LABELS[n] for n in names])
    ax.set_ylabel("wall-clock seconds")
    ax.grid(True, axis="y")
    for i, (name, value) in enumerate(zip(names, vals)):
        if name == "retrain" or not baseline:
            label = f"{value:.0f} s\nbaseline"
        else:
            label = f"{value:.0f} s\n{baseline / value:.1f}x"
        ax.text(
            i,
            value,
            label,
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
    ax.margins(y=0.22)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.18)
    return _save(fig, path)


def attention_hist(per_model: dict, path: str) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for name in MODEL_ORDER:
        vals = per_model.get(name)
        if vals is None or len(vals) == 0:
            continue
        ax.hist(
            np.asarray(vals),
            bins=20,
            histtype="step",
            linewidth=1.2,
            linestyle=MODEL_LINESTYLES[name],
            label=MODEL_LABELS[name],
            color=MODEL_COLORS[name],
        )
    ax.set_xlabel("fraction of Grad-CAM mass on trigger")
    ax.set_ylabel("number of images")
    ax.grid(True, axis="y")
    ax.legend()
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.19)
    return _save(fig, path)


def _square_png(arr: np.ndarray, path: str, factor: int = 12) -> str:
    image = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    image = image.resize(
        (image.width * factor, image.height * factor), resample=Image.NEAREST
    )
    image.save(path)
    return path


def hero_tiles(
    models: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    out_dir: str,
    device: str = "cpu",
) -> dict:
    """Pick one triggered sample and write text-free square PNGs for input + overlays."""
    ensure_dir(out_dir)
    y = y.to(x.device)
    rows = _pred_rows(models, x, device)
    orig = rows.get("original")
    pga = rows.get("pga")

    index = 0
    if orig is not None and pga is not None:
        for j in range(x.shape[0]):
            if int(orig[1][j]) == 0 and int(pga[1][j]) == int(y[j].item()):
                index = j
                break

    cpu_x = x.detach().cpu()
    img = cpu_x[index].numpy().transpose(1, 2, 0)
    _square_png(img, os.path.join(out_dir, "hero_input.png"))
    for name in MODEL_ORDER:
        if name not in rows:
            continue
        cams, _, _ = rows[name]
        overlay = gradcam.overlay(img, cams[index]) / 255.0
        _square_png(overlay, os.path.join(out_dir, f"hero_{name}.png"))

    true_idx = int(y[index].item())
    result = {
        "true": CLASSES[true_idx] if true_idx < len(CLASSES) else str(true_idx),
        "index": index,
        "models": {},
    }
    for name in MODEL_ORDER:
        if name not in rows:
            continue
        _, preds, probs = rows[name]
        pred_idx = int(preds[index])
        result["models"][name] = {
            "pred": CLASSES[pred_idx] if pred_idx < len(CLASSES) else str(pred_idx),
            "confidence": float(probs[index]),
        }
    return result


def trigger_closeup(path: str) -> str:
    """Write the 6x6 trigger pattern upscaled x40, with no text."""
    from .data import _trigger_pattern

    pattern = _trigger_pattern().numpy().transpose(1, 2, 0)
    image = Image.fromarray((np.clip(pattern, 0, 1) * 255).astype(np.uint8))
    image = image.resize((image.width * 40, image.height * 40), resample=Image.NEAREST)
    image.save(path)
    return path
