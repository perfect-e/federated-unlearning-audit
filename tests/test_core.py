from __future__ import annotations

import numpy as np
import pytest
import torch

from fedunlearn import audit, data, gradcam, unlearn
from fedunlearn.fl import fedavg
from fedunlearn.model import ResNetSmall


def test_partition_iid_covers_all_indices_once():
    n, clients = 103, 5
    parts = data.partition_iid(n, clients, seed=3)
    joined = torch.cat(parts)
    assert len(parts) == clients
    assert joined.numel() == n
    assert torch.equal(torch.sort(joined).values, torch.arange(n))


def test_partition_dirichlet_covers_all_indices_once():
    labels = torch.tensor([i % 4 for i in range(200)])
    parts = data.partition_dirichlet(labels.numpy(), 6, alpha=0.5, seed=1)
    joined = torch.cat(parts)
    assert joined.numel() == labels.numel()
    assert torch.equal(torch.sort(joined).values, torch.arange(labels.numel()))
    assert all(len(p) > 0 for p in parts)


def test_trigger_stamps_only_corner_region():
    x = torch.zeros(4, 3, 32, 32)
    stamped = data.stamp_trigger(x)
    mask = data.trigger_mask(32)
    assert stamped.shape == x.shape
    assert torch.equal(stamped[:, :, ~mask], x[:, :, ~mask])
    assert not torch.equal(stamped[:, :, mask], x[:, :, mask])
    assert float(stamped.min()) >= 0.0
    assert float(stamped.max()) <= 1.0


def test_poison_mask_is_deterministic_and_sized():
    m1 = data.make_poison_mask(100, 0.8, seed=5)
    m2 = data.make_poison_mask(100, 0.8, seed=5)
    assert torch.equal(m1, m2)
    assert int(m1.sum()) == 80


def test_fedavg_identical_models_is_identity():
    base = {
        "w": torch.randn(3, 3),
        "b": torch.randn(3),
        "count": torch.tensor(7, dtype=torch.long),
    }
    clones = [{k: v.clone() for k, v in base.items()} for _ in range(3)]
    avg = fedavg(clones, [1.0, 2.0, 4.0])
    assert torch.allclose(avg["w"], base["w"])
    assert torch.allclose(avg["b"], base["b"])
    assert avg["count"].dtype == torch.long
    assert int(avg["count"]) == 7


def test_fedavg_weighted_average_toy_case():
    s1 = {"w": torch.zeros(2), "n": torch.tensor(1, dtype=torch.long)}
    s2 = {"w": torch.tensor([4.0, 8.0]), "n": torch.tensor(2, dtype=torch.long)}
    avg = fedavg([s1, s2], [1.0, 3.0])
    expected = (0.0 * 1 + 4.0 * 3) / 4.0
    expected2 = (0.0 * 1 + 8.0 * 3) / 4.0
    assert torch.allclose(avg["w"], torch.tensor([expected, expected2]))


def test_pga_projection_keeps_distance_within_radius():
    reference = [torch.zeros(5)]
    params = [torch.nn.Parameter(torch.full((5,), 10.0))]
    radius = 3.0
    unlearn.project_to_ball(params, reference, radius)
    dist = float(torch.norm(params[0].detach() - reference[0]))
    assert dist <= radius + 1e-4
    assert dist == pytest.approx(radius, rel=1e-4)


def test_projection_leaves_small_moves_unchanged():
    reference = [torch.zeros(4)]
    params = [torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.1, 0.1]))]
    before = params[0].detach().clone()
    unlearn.project_to_ball(params, reference, 5.0)
    assert torch.allclose(params[0].detach(), before)


def test_gradcam_output_shape_and_range():
    torch.manual_seed(0)
    model = ResNetSmall()
    x = torch.rand(2, 3, 32, 32)
    cam, logits, cls = gradcam.compute_cam(model, x, "cpu")
    assert cam.shape == (2, 8, 8)
    assert logits.shape == (2, 10)
    assert cls.shape == (2,)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0 + 1e-6
    up = gradcam.upsample(cam, 32)
    assert up.shape == (2, 32, 32)


def test_gradcam_overlay_is_rgb_uint8():
    img = np.random.rand(32, 32, 3)
    cam = np.random.rand(32, 32)
    out = gradcam.overlay(img, cam)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_auc_rank_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert audit.roc_auc_rank(scores, labels) == pytest.approx(1.0)


def test_auc_rank_inverted_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1, 1, 0, 0])
    assert audit.roc_auc_rank(scores, labels) == pytest.approx(0.0)


def test_auc_rank_ties_are_chance():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])
    assert audit.roc_auc_rank(scores, labels) == pytest.approx(0.5)


def test_config_serialises_to_json():
    import json

    from fedunlearn.config import Config

    for cfg in (Config(), Config.quick()):
        json.dumps(cfg.to_dict())
    assert Config().is_quick is False and Config.quick().is_quick is True


def _verdict_metrics(original_auc: float, mia_value: float = 0.9) -> dict:
    return {
        "original": {"mia_auc": original_auc},
        "retrain": {
            "asr": 0.02,
            "test_acc": 0.81,
            "mia_auc": 0.50,
            "trigger_attention": 0.03,
        },
        "finetune": {
            "asr": 0.02,
            "test_acc": 0.81,
            "mia_auc": mia_value,
            "trigger_attention": 0.03,
        },
        "pga": {
            "asr": 0.02,
            "test_acc": 0.81,
            "mia_auc": mia_value,
            "trigger_attention": 0.03,
        },
    }


def test_mia_check_is_uninformative_when_original_auc_is_chance():
    metrics = _verdict_metrics(original_auc=0.496)
    verdicts = audit.build_verdicts(metrics)
    for name in ("finetune", "pga"):
        assert verdicts[name]["mia_auc"]["informative"] is False
        for check in ("asr", "test_acc", "trigger_attention"):
            assert verdicts[name][check]["informative"] is True
        assert verdicts[name]["all_passed"] is True
        assert verdicts[name]["passed_count"] == 3
        assert verdicts[name]["informative_count"] == 3


def test_mia_check_is_informative_when_original_auc_differs_from_chance():
    metrics = _verdict_metrics(original_auc=0.40)
    verdicts = audit.build_verdicts(metrics)
    for name in ("finetune", "pga"):
        assert verdicts[name]["mia_auc"]["informative"] is True
        assert verdicts[name]["all_passed"] is False
        assert verdicts[name]["passed_count"] == 3
        assert verdicts[name]["informative_count"] == 4


def test_build_index_groups_sorts_and_handles_see(tmp_path):
    from fedunlearn.latex_report import build_index

    idx = tmp_path / "main.idx"
    idx.write_text(
        "\\indexentry{Grad-CAM|hyperpage}{3}\n"
        "\\indexentry{Grad-CAM!difference maps|hyperpage}{12}\n"
        "\\indexentry{backdoor|hyperpage}{2}\n"
        "\\indexentry{backdoor|hyperpage}{2}\n"
        "\\indexentry{backdoor|hyperpage}{9}\n"
        "\\indexentry{efficiency|hyperindexformat{\\see{cost}}}{3}\n",
        encoding="utf-8",
    )
    ind = tmp_path / "main.ind"
    assert build_index(str(idx), str(ind)) == 3
    lines = ind.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "\\ixletter{B}",
        "\\ixitem{backdoor}{\\hyperpage{2}, \\hyperpage{9}}",
        "\\ixletter{E}",
        "\\ixsee{efficiency}{cost}",
        "\\ixletter{G}",
        "\\ixitem{Grad-CAM}{\\hyperpage{3}}",
        "\\ixsub{difference maps}{\\hyperpage{12}}",
    ]


def test_device_override_from_environment(monkeypatch):
    from fedunlearn import config

    monkeypatch.setenv("FEDUNLEARN_DEVICE", "cpu")
    assert config.default_device() == "cpu"
    assert config.Config().device == "cpu"


def test_offline_tex_flag(monkeypatch):
    from fedunlearn import latex_report

    monkeypatch.setattr(latex_report.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.delenv("FEDUNLEARN_TEX_OFFLINE", raising=False)
    assert "--only-cached" not in latex_report._engine_command()
    monkeypatch.setenv("FEDUNLEARN_TEX_OFFLINE", "1")
    assert "--only-cached" in latex_report._engine_command()
