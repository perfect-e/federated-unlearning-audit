from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from fedunlearn import audit, data, fl, latex_report, unlearn, viz
from fedunlearn.config import Config, environment, set_seed
from fedunlearn.model import ResNetSmall

CHECKPOINTS = ("init", "original", "retrain", "finetune", "pga")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated unlearning demo with auditing and visualisation"
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--poison-frac", type=float, default=None)
    parser.add_argument("--recovery-rounds", type=int, default=None)
    parser.add_argument("--pga-radius-frac", type=float, default=None)
    parser.add_argument("--pga-lr", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="compute device (default: FEDUNLEARN_DEVICE, else mps, cuda, cpu)",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "train", "unlearn", "audit", "report"],
        default="all",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="write LaTeX sources and figures but skip compiling the PDF",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.quick() if args.quick else Config()
    overrides = {
        "rounds": args.rounds,
        "num_clients": args.clients,
        "seed": args.seed,
        "alpha": args.alpha,
        "poison_frac": args.poison_frac,
        "recovery_rounds": args.recovery_rounds,
        "pga_radius_frac": args.pga_radius_frac,
        "pga_lr": args.pga_lr,
        "out_dir": args.out,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if cfg.num_clients < 2:
        raise SystemExit("--clients must be at least 2 (one client is forgotten)")
    return cfg


def ckpt_path(cfg: Config, name: str) -> str:
    return os.path.join(cfg.out_dir, "checkpoints", f"{name}.pt")


def history_path(cfg: Config, name: str) -> str:
    return os.path.join(cfg.out_dir, "checkpoints", f"{name}_history.json")


def have_checkpoint(cfg: Config, name: str) -> bool:
    return os.path.exists(ckpt_path(cfg, name))


def save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_state(path: str, device: str) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    return payload["state"]


def save_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def prepare(cfg: Config):
    set_seed(cfg.seed)
    print(f"device: {cfg.device} | out: {cfg.out_dir}")
    tx, ty, vx, vy = data.load_cifar10(
        cfg.data_dir, cfg.device, cfg.train_subset, cfg.test_subset, cfg.seed
    )
    if cfg.alpha is None:
        indices = data.partition_iid(tx.shape[0], cfg.num_clients, cfg.seed)
    else:
        indices = data.partition_dirichlet(
            ty.detach().cpu().numpy(), cfg.num_clients, cfg.alpha, cfg.seed
        )
    clients = data.build_clients(
        tx, ty, indices, cfg.poison_frac, cfg.seed, cfg.forget_client
    )
    trig_x, trig_y = data.triggered_test_set(vx, vy)
    global_eval = fl.make_eval_fn(vx, vy, trig_x, trig_y, cfg.device, cfg.eval_batch)

    rng = np.random.default_rng(cfg.seed + 7)
    retain_pool = clients[1:]
    total_pool = sum(c.num_samples for c in retain_pool)
    retain_count = min(cfg.retain_samples, total_pool)
    retain_x = []
    retain_y = []
    for client in retain_pool:
        share = retain_count * client.num_samples / max(total_pool, 1)
        take = min(client.num_samples, int(round(share)))
        if take <= 0:
            continue
        idx = rng.choice(client.num_samples, size=take, replace=False)
        retain_x.append(client.x[idx])
        retain_y.append(client.y[idx])
    retain_x = torch.cat(retain_x)[:retain_count]
    retain_y = torch.cat(retain_y)[:retain_count]

    forget_x, forget_y = data.forget_eval_set(clients[cfg.forget_client])
    nonmember_x, nonmember_y = data.nonmember_set(
        vx, vy, forget_x.shape[0], cfg.poison_frac, cfg.seed
    )

    return {
        "train_x": tx,
        "train_y": ty,
        "clean_x": vx,
        "clean_y": vy,
        "trig_x": trig_x,
        "trig_y": trig_y,
        "clients": clients,
        "eval_fn": global_eval,
        "retain_x": retain_x,
        "retain_y": retain_y,
        "forget_x": forget_x,
        "forget_y": forget_y,
        "nonmember_x": nonmember_x,
        "nonmember_y": nonmember_y,
    }


def stage_train(cfg: Config, ctx: dict, force: bool) -> None:
    if have_checkpoint(cfg, "original") and have_checkpoint(cfg, "init") and not force:
        print("[train] checkpoints present, skipping")
        return
    print("[train] federated training over all clients")
    init_model = ResNetSmall().to(cfg.device)
    init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
    save_checkpoint(ckpt_path(cfg, "init"), {"state": init_state})

    model = ResNetSmall().to(cfg.device)
    model.load_state_dict(init_state)
    t0 = time.time()
    model, history = fl.run_federated(
        model,
        ctx["clients"],
        cfg.rounds,
        device=cfg.device,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        local_epochs=cfg.local_epochs,
        eval_fn=ctx["eval_fn"],
    )
    seconds = time.time() - t0
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    save_checkpoint(
        ckpt_path(cfg, "original"),
        {
            "state": state,
            "history": history,
            "seconds": seconds,
            "device": cfg.device,
            "rounds": cfg.rounds,
            "client_rounds": cfg.rounds * cfg.num_clients,
        },
    )
    save_json(history_path(cfg, "original"), history)
    print(f"[train] done in {seconds:.1f}s")


def stage_unlearn(cfg: Config, ctx: dict, force: bool) -> None:
    clients = ctx["clients"]
    total_rounds = cfg.rounds
    if not have_checkpoint(cfg, "init"):
        stage_train(cfg, ctx, force)
    init_state = load_state(ckpt_path(cfg, "init"), cfg.device)
    if not have_checkpoint(cfg, "original"):
        stage_train(cfg, ctx, force)

    if have_checkpoint(cfg, "retrain") and not force:
        print("[unlearn] retrain present, skipping")
    else:
        print("[unlearn] retrain (reference)")
        model, history, secs = unlearn.retrain(
            init_state,
            clients,
            cfg.rounds,
            cfg.device,
            eval_fn=ctx["eval_fn"],
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            local_epochs=cfg.local_epochs,
        )
        save_checkpoint(
            ckpt_path(cfg, "retrain"),
            {
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "history": history,
                "seconds": secs,
                "device": cfg.device,
                "client_rounds": cfg.rounds * (cfg.num_clients - 1),
            },
        )
        save_json(history_path(cfg, "retrain"), history)
        print(f"[unlearn] retrain done in {secs:.1f}s")

    original_model = ResNetSmall().to(cfg.device)
    original_model.load_state_dict(load_state(ckpt_path(cfg, "original"), cfg.device))

    if have_checkpoint(cfg, "finetune") and not force:
        print("[unlearn] finetune present, skipping")
    else:
        print("[unlearn] finetune (naive baseline)")
        model, history, secs = unlearn.finetune(
            original_model,
            clients,
            cfg.recovery_rounds,
            cfg.device,
            eval_fn=ctx["eval_fn"],
            lr=cfg.finetune_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            local_epochs=cfg.local_epochs,
            base_round=total_rounds,
        )
        save_checkpoint(
            ckpt_path(cfg, "finetune"),
            {
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "history": history,
                "seconds": secs,
                "device": cfg.device,
                "client_rounds": cfg.recovery_rounds * (cfg.num_clients - 1),
            },
        )
        save_json(history_path(cfg, "finetune"), history)
        print(f"[unlearn] finetune done in {secs:.1f}s")

    if have_checkpoint(cfg, "pga") and not force:
        print("[unlearn] pga present, skipping")
    else:
        print("[unlearn] pga (projected gradient ascent + recovery)")
        init_model = ResNetSmall().to(cfg.device)
        init_model.load_state_dict(init_state)
        model, history, secs = unlearn.pga(
            original_model,
            init_model,
            clients[cfg.forget_client],
            clients,
            cfg.recovery_rounds,
            cfg.device,
            eval_fn=ctx["eval_fn"],
            pga_lr=cfg.pga_lr,
            pga_radius_frac=cfg.pga_radius_frac,
            pga_epochs=cfg.pga_epochs,
            pga_target_acc=cfg.pga_target_acc,
            pga_clip=cfg.pga_clip,
            recovery_lr=cfg.finetune_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            local_epochs=cfg.local_epochs,
            base_round=total_rounds,
        )
        save_checkpoint(
            ckpt_path(cfg, "pga"),
            {
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "history": history,
                "seconds": secs,
                "device": cfg.device,
                "client_rounds": cfg.recovery_rounds * (cfg.num_clients - 1) + 1,
            },
        )
        save_json(history_path(cfg, "pga"), history)
        print(f"[unlearn] pga done in {secs:.1f}s")


def load_models(cfg: Config) -> dict:
    models = {}
    for name in ("original", "retrain", "finetune", "pga"):
        path = ckpt_path(cfg, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing checkpoint {path}; run the train/unlearn stages first")
        model = ResNetSmall().to(cfg.device)
        model.load_state_dict(load_state(path, cfg.device))
        model.eval()
        models[name] = model
    return models


def training_device(cfg: Config) -> str:
    """Device the checkpoints were trained on, which may differ from the audit device."""
    saved = torch.load(ckpt_path(cfg, "original"), map_location="cpu", weights_only=False)
    if saved.get("device"):
        return saved["device"]
    previous = os.path.join(cfg.out_dir, "metrics.json")
    if os.path.exists(previous):
        return load_json(previous).get("config", {}).get("device", cfg.device)
    return cfg.device


def stage_audit(cfg: Config, ctx: dict, force: bool) -> dict:
    print("[audit] scoring all models")
    models = load_models(cfg)
    costs = {}
    histories = {}
    for name in ("original", "retrain", "finetune", "pga"):
        payload = torch.load(ckpt_path(cfg, name), map_location="cpu", weights_only=False)
        costs[name] = {
            "seconds": float(payload.get("seconds", 0.0)),
            "rounds": int(payload.get("client_rounds", 0)),
        }
        hist_file = history_path(cfg, name)
        histories[name] = load_json(hist_file) if os.path.exists(hist_file) else payload.get("history", [])
    metrics = audit.audit_all(
        models,
        clean_x=ctx["clean_x"],
        clean_y=ctx["clean_y"],
        trig_x=ctx["trig_x"],
        trig_y=ctx["trig_y"],
        forget_x=ctx["forget_x"],
        forget_y=ctx["forget_y"],
        nonmember_x=ctx["nonmember_x"],
        nonmember_y=ctx["nonmember_y"],
        retain_x=ctx["retain_x"],
        retain_y=ctx["retain_y"],
        device=cfg.device,
        cam_samples=cfg.cam_samples,
        costs=costs,
        per_image_attention=True,
    )
    verdicts = audit.build_verdicts(metrics)
    config = cfg.to_dict()
    config["device"] = training_device(cfg)
    env = environment()
    env["audit_device"] = cfg.device
    payload = {
        "config": config,
        "environment": env,
        "metrics": metrics,
        "verdicts": verdicts,
        "histories": histories,
    }
    save_json(os.path.join(cfg.out_dir, "metrics.json"), payload)
    return payload


def stage_report(cfg: Config, ctx: dict, payload: dict, compile_pdf: bool = True) -> str:
    print("[report] building figures and LaTeX sources")
    models = load_models(cfg)
    fig_dir = os.path.join(cfg.out_dir, "figures")
    viz.ensure_dir(fig_dir)

    metrics = payload["metrics"]
    histories = payload["histories"]

    viz.training_curves(
        metrics, histories, cfg.rounds, os.path.join(fig_dir, "training_curves.png")
    )
    viz.metrics_bars(metrics, os.path.join(fig_dir, "metrics_bars.png"))

    rng = np.random.default_rng(cfg.seed + 11)
    trig_count = min(8, ctx["trig_x"].shape[0])
    trig_idx = np.sort(rng.choice(ctx["trig_x"].shape[0], trig_count, replace=False))
    trig_sel_x = ctx["trig_x"][torch.from_numpy(trig_idx)]
    trig_sel_y = ctx["trig_y"][torch.from_numpy(trig_idx)]

    clean_count = min(8, ctx["clean_x"].shape[0])
    clean_idx = np.sort(rng.choice(ctx["clean_x"].shape[0], clean_count, replace=False))
    clean_sel_x = ctx["clean_x"][torch.from_numpy(clean_idx)]
    clean_sel_y = ctx["clean_y"][torch.from_numpy(clean_idx)]

    viz.gradcam_grid(
        models,
        trig_sel_x,
        trig_sel_y,
        os.path.join(fig_dir, "gradcam_triggered.png"),
        device=cfg.device,
    )
    viz.gradcam_grid(
        models,
        clean_sel_x,
        clean_sel_y,
        os.path.join(fig_dir, "gradcam_clean.png"),
        device=cfg.device,
    )
    viz.attention_diff(
        models,
        trig_sel_x,
        trig_sel_y,
        os.path.join(fig_dir, "attention_diff.png"),
        device=cfg.device,
    )
    viz.cost_chart(metrics, os.path.join(fig_dir, "cost.png"))

    per_model_attention = {}
    for name in metrics:
        per_model_attention[name] = metrics[name].get(
            "trigger_attention_per_image", []
        )
    viz.attention_hist(
        per_model_attention, os.path.join(fig_dir, "trigger_attention_hist.png")
    )

    hero = viz.hero_tiles(models, trig_sel_x, trig_sel_y, fig_dir, device=cfg.device)
    viz.trigger_closeup(os.path.join(fig_dir, "trigger_patch.png"))

    save_json(os.path.join(cfg.out_dir, "metrics.json"), payload)
    ctx_info = {
        "train_size": int(ctx["train_x"].shape[0]),
        "test_size": int(ctx["clean_x"].shape[0]),
        "triggered_test_size": int(ctx["trig_x"].shape[0]),
        "forget_set_size": int(ctx["forget_x"].shape[0]),
        "poisoned_count": int(ctx["clients"][cfg.forget_client].poison_mask.sum()),
        "cam_samples": cfg.cam_samples,
        "model_params": int(
            sum(p.numel() for p in ResNetSmall().parameters() if p.requires_grad)
        ),
        "hero": hero,
    }
    template_dir = os.environ.get("FEDUNLEARN_TEMPLATE_DIR", "report_template")
    result = latex_report.build(
        payload,
        cfg,
        ctx_info,
        cfg.out_dir,
        template_dir=template_dir,
        compile_pdf=compile_pdf,
    )
    if result.get("pdf"):
        print(f"[report] wrote {result['pdf']}")
        return result["pdf"]
    print(f"[report] wrote LaTeX sources to {result['report_dir']}")
    return result["report_dir"]


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    ctx = prepare(cfg)
    os.makedirs(os.path.join(cfg.out_dir, "checkpoints"), exist_ok=True)

    payload = None
    if args.stage in ("all", "train", "unlearn"):
        stage_train(cfg, ctx, args.force)
    if args.stage in ("all", "unlearn"):
        stage_unlearn(cfg, ctx, args.force)
    if args.stage in ("all", "audit", "report"):
        payload = stage_audit(cfg, ctx, args.force)
    if args.stage in ("all", "report"):
        stage_report(cfg, ctx, payload, compile_pdf=not args.no_pdf)

    if payload is not None:
        print("\n=== summary ===")
        for name, m in payload["metrics"].items():
            print(
                f"{name:9s} acc {m['test_acc']:.4f}  asr {m['asr']:.4f}  "
                f"forget {m['forget_acc']:.4f}  mia {m['mia_auc']:.3f}  "
                f"attn {m['trigger_attention']:.3f}"
            )
        for name, v in payload["verdicts"].items():
            state = "PASS" if v.get("all_passed") else "FAIL"
            print(f"verdict {name}: {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
