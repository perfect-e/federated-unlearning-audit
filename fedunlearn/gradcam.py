from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import normalize


class _HookCapture:
    def __init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

    def attach(self, layer: nn.Module) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0]

        self._handles.append(layer.register_forward_hook(forward_hook))
        self._handles.append(layer.register_full_backward_hook(backward_hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def compute_cam(
    model: nn.Module,
    x_pixel: torch.Tensor,
    device: str = "cpu",
    target_class: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard Grad-CAM for the predicted (or given) class.

    Returns (cam, logits, classes) where cam has shape (B, 8, 8) in [0, 1].
    """
    was_training = model.training
    model.eval()
    capture = _HookCapture()
    capture.attach(model.cam_layer)
    try:
        x = normalize(x_pixel).to(device)
        logits = model(x)
        classes = target_class if target_class is not None else logits.argmax(1)
        classes = classes.to(logits.device)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, classes.view(-1, 1), 1.0)
        model.zero_grad(set_to_none=True)
        logits.backward(gradient=one_hot)
        acts = capture.activations.detach()
        grads = capture.gradients.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        b = cam.shape[0]
        cam_flat = cam.view(b, -1)
        cmin = cam_flat.min(dim=1, keepdim=True).values
        cmax = cam_flat.max(dim=1, keepdim=True).values
        denom = (cmax - cmin).clamp_min(1e-8)
        cam = ((cam_flat - cmin) / denom).view_as(cam).squeeze(1)
    finally:
        capture.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return cam.detach().cpu(), logits.detach().cpu(), classes.detach().cpu()


def upsample(cam: torch.Tensor, size: int = 32) -> torch.Tensor:
    if cam.dim() == 2:
        cam = cam.unsqueeze(0)
    up = F.interpolate(cam.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False)
    return up.squeeze(1)


def _colormap(name: str = "jet") -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps[name]
    return cmap(np.linspace(0, 1, 256))[:, :3]


def overlay(img: np.ndarray, cam: np.ndarray, colormap: str = "jet", alpha: float = 0.5) -> np.ndarray:
    """Blend a CAM heatmap over an RGB image (both in pixel space, [0, 1])."""
    img = np.clip(img, 0, 1)
    cam = np.clip(cam, 0, 1)
    lut = _colormap(colormap)
    idx = np.clip((cam * 255).astype(np.int64), 0, 255)
    heat = lut[idx]
    blended = (1 - alpha) * img + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def cam_for_batch(
    model: nn.Module, x_pixel: torch.Tensor, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (upsampled cams as numpy (B,32,32), preds, probs)."""
    cam, logits, preds = compute_cam(model, x_pixel, device)
    up = upsample(cam, x_pixel.shape[-1]).clamp(0, 1).numpy()
    probs = torch.softmax(logits, dim=1).gather(1, preds.view(-1, 1)).squeeze(1).numpy()
    return up, preds.numpy(), probs
