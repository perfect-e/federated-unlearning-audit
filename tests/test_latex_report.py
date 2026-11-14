from __future__ import annotations

import re
from pathlib import Path

import pytest

from fedunlearn import latex_report
from fedunlearn.config import Config


LABELS = ["Original", "Finetune", "Pga", "Retrain"]


def _metrics() -> dict:
    return {
        "original": {
            "test_acc": 0.8022,
            "asr": 0.9973,
            "forget_acc": 0.9618,
            "retain_acc": 0.8348,
            "mia_auc": 0.4960,
            "trigger_attention": 0.3689,
            "trigger_attention_baseline": 0.0625,
            "cost_seconds": 1156.2,
            "cost_rounds": 300,
        },
        "retrain": {
            "test_acc": 0.8120,
            "asr": 0.0244,
            "forget_acc": 0.2494,
            "retain_acc": 0.8460,
            "mia_auc": 0.5002,
            "trigger_attention": 0.0326,
            "trigger_attention_baseline": 0.0625,
            "cost_seconds": 1123.3,
            "cost_rounds": 270,
        },
        "finetune": {
            "test_acc": 0.8115,
            "asr": 0.5693,
            "forget_acc": 0.6504,
            "retain_acc": 0.8454,
            "mia_auc": 0.4964,
            "trigger_attention": 0.2135,
            "trigger_attention_baseline": 0.0625,
            "cost_seconds": 187.6,
            "cost_rounds": 45,
        },
        "pga": {
            "test_acc": 0.8069,
            "asr": 0.0373,
            "forget_acc": 0.2574,
            "retain_acc": 0.8426,
            "mia_auc": 0.4990,
            "trigger_attention": 0.0430,
            "trigger_attention_baseline": 0.0625,
            "cost_seconds": 204.9,
            "cost_rounds": 46,
        },
    }


def _payload() -> dict:
    from fedunlearn import audit

    metrics = _metrics()
    return {
        "config": Config().to_dict(),
        "metrics": metrics,
        "verdicts": audit.build_verdicts(metrics),
        "histories": {
            "original": [
                {"round": 1, "test_acc": 0.2, "asr": 0.1},
                {"round": 2, "test_acc": 0.8, "asr": 0.99},
            ],
            "retrain": [
                {"round": 1, "test_acc": 0.21, "asr": 0.06},
                {"round": 2, "test_acc": 0.81, "asr": 0.02},
            ],
            "finetune": [{"round": 3, "test_acc": 0.81, "asr": 0.56}],
            "pga": [
                {
                    "round": 2,
                    "test_acc": 0.48,
                    "asr": 0.0,
                    "seconds": 4.6,
                    "phase": "ascent",
                    "forget_acc": 0.1,
                    "pga_epochs": 1,
                },
                {
                    "round": 3,
                    "test_acc": 0.807,
                    "asr": 0.03,
                    "seconds": 40.0,
                    "phase": "recovery",
                },
            ],
        },
    }


def _ctx_info() -> dict:
    return {
        "train_size": 50000,
        "test_size": 10000,
        "triggered_test_size": 9000,
        "forget_set_size": 5000,
        "poisoned_count": 4000,
        "cam_samples": 500,
        "model_params": 696618,
        "hero": {
            "true": "frog",
            "models": {
                "original": {"pred": "airplane", "confidence": 0.97},
                "finetune": {"pred": "frog", "confidence": 0.61},
                "pga": {"pred": "frog", "confidence": 0.88},
                "retrain": {"pred": "frog", "confidence": 0.9},
            },
        },
    }


def _macro_names() -> list[str]:
    names = []
    for label in LABELS:
        names += [
            f"TestAcc{label}",
            f"Asr{label}",
            f"ForgetAcc{label}",
            f"RetainAcc{label}",
            f"Mia{label}",
            f"Attn{label}",
            f"CostSec{label}",
            f"CostMin{label}",
            f"ClientRounds{label}",
            f"HeroPred{label}",
            f"HeroConf{label}",
        ]
    names += [
        "AttnBaseline",
        "SpeedupPga",
        "SpeedupFinetune",
        "AsrDropPga",
        "AsrGapPga",
        "NumClients",
        "NumRetained",
        "NumRounds",
        "RecoveryRounds",
        "PoisonPct",
        "Seed",
        "DeviceName",
        "BatchSize",
        "LocalEpochs",
        "BaseLr",
        "FinetuneLr",
        "PgaLr",
        "PgaRadiusFrac",
        "PgaClip",
        "PgaTargetAccPct",
        "Partition",
        "TrainSize",
        "TestSize",
        "TriggeredTestSize",
        "ForgetSetSize",
        "PoisonedCount",
        "CamSamples",
        "ModelParams",
        "RunDate",
        "PgaEpochsUsed",
        "PgaAscentTestAcc",
        "PgaAscentAsr",
        "PgaAscentForgetAcc",
        "PgaAscentSec",
        "ChecksPassedFinetune",
        "ChecksPassedPga",
        "ChecksInformative",
        "HeroTrue",
    ]
    for verdict in ("Finetune", "Pga"):
        for check in ("Asr", "Acc", "Mia", "Attn"):
            names.append(f"Verdict{verdict}{check}")
    return names


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "main.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
    out_dir = tmp_path / "outputs"
    figures = out_dir / "figures"
    figures.mkdir(parents=True)
    (figures / "training_curves.png").write_bytes(b"png")
    return template_dir, out_dir, figures


def test_data_tex_macros_are_complete_and_unique(tmp_path):
    template_dir, out_dir, figures = _prepare(tmp_path)
    result = latex_report.build(
        _payload(),
        Config(),
        _ctx_info(),
        str(out_dir),
        template_dir=str(template_dir),
        compile_pdf=False,
    )
    text = Path(result["data_tex"]).read_text(encoding="utf-8")

    for name in _macro_names():
        pattern = re.compile(r"\\newcommand\{\\" + re.escape(name) + r"\}")
        assert len(pattern.findall(text)) == 1, name

    for name in ("MiaInformative", "PgaAllPassed", "FinetuneAllPassed"):
        pattern = re.compile(r"\\newif\\if" + re.escape(name))
        assert len(pattern.findall(text)) == 1, name

    assert not re.search(r"(?<!\\)%", text)
    verdict_pattern = re.compile(
        r"\\newcommand\{\\Verdict(?:Finetune|Pga)(?:Asr|Acc|Mia|Attn)\}"
        r"\{(?:\\PassMark|\\FailMark|\\NAMark)\}"
    )
    assert len(verdict_pattern.findall(text)) == 8
    assert r"\VerdictFinetuneMia}{\NAMark}" in text
    assert r"\VerdictPgaAsr}{\PassMark}" in text
    assert r"\MiaInformativefalse" in text
    assert r"\PgaAllPassedtrue" in text
    assert r"\FinetuneAllPassedfalse" in text
    assert r"\ModelParams}{696{,}618}" in text
    assert r"\HeroTrue}{frog}" in text
    assert r"\HeroPredOriginal}{airplane}" in text
    assert r"\HeroConfOriginal}{97}" in text

    assert (out_dir / "report" / "main.tex").is_file()
    assert (out_dir / "report" / "figures" / "training_curves.png").is_file()


def test_history_and_recovery_rows(tmp_path):
    template_dir, out_dir, _ = _prepare(tmp_path)
    result = latex_report.build(
        _payload(),
        Config(),
        _ctx_info(),
        str(out_dir),
        template_dir=str(template_dir),
        compile_pdf=False,
    )
    history = Path(result["history_rows"]).read_text(encoding="utf-8").splitlines()
    assert len(history) == 2
    assert history[0].startswith("1 & 20.0 & 10.0 & 21.0 & 6.0")
    assert history[0].endswith(r"\\")

    recovery = Path(result["recovery_rows"]).read_text(encoding="utf-8").splitlines()
    assert len(recovery) == 2
    assert recovery[0].startswith("Ascent & -- & -- & 48.0 & 0.0")
    assert recovery[1].startswith("T+1 & 81.0 & 56.0 & 80.7 & 3.0")


def test_missing_template_is_a_clear_error(tmp_path):
    out_dir = tmp_path / "outputs"
    (out_dir / "figures").mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        latex_report.build(
            _payload(),
            Config(),
            _ctx_info(),
            str(out_dir),
            template_dir=str(tmp_path / "does-not-exist"),
            compile_pdf=False,
        )
    assert "LaTeX template not found" in str(exc.value)
