from __future__ import annotations

import datetime
import os
import re
import shutil
import subprocess

MODEL_MACROS = {
    "original": "Original",
    "finetune": "Finetune",
    "pga": "Pga",
    "retrain": "Retrain",
}
VERDICT_MODELS = {"Finetune": "finetune", "Pga": "pga"}
VERDICT_CHECKS = {
    "Asr": "asr",
    "Acc": "test_acc",
    "Mia": "mia_auc",
    "Attn": "trigger_attention",
}
MIA_TOL = 0.02


def _esc(text) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _pct(value) -> str:
    return f"{float(value) * 100:.1f}"


def _seconds(value) -> str:
    return f"{int(round(float(value)))}"


def _minutes(value) -> str:
    return f"{float(value) / 60.0:.1f}"


def _plain_float(value) -> str:
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:g}"


def _partition(config: dict) -> str:
    alpha = config.get("alpha")
    if alpha is None:
        return "IID"
    return f"Dirichlet($\\alpha$={_plain_float(alpha)})"


def _run_date(out_dir: str, ctx_info: dict) -> str:
    explicit = ctx_info.get("run_date")
    if explicit:
        return str(explicit)
    checkpoint = os.path.join(out_dir, "checkpoints", "original.pt")
    if os.path.isfile(checkpoint):
        moment = datetime.datetime.fromtimestamp(os.path.getmtime(checkpoint))
    else:
        moment = datetime.datetime.now()
    return f"{moment.day} {moment.strftime('%B')} {moment.year}"


def _model_params(ctx_info: dict) -> int:
    explicit = ctx_info.get("model_params")
    if explicit is not None:
        return int(explicit)
    from .model import ResNetSmall

    return int(sum(p.numel() for p in ResNetSmall().parameters() if p.requires_grad))


def _check_informative(name: str, check: str, verdicts: dict, metrics: dict) -> bool:
    entry = (verdicts.get(name) or {}).get(check)
    if isinstance(entry, dict) and "informative" in entry:
        return bool(entry["informative"])
    if check == "mia_auc":
        original_auc = float((metrics.get("original") or {}).get("mia_auc", 0.5))
        return abs(original_auc - 0.5) > MIA_TOL
    return True


def _verdict_mark(name: str, check: str, verdicts: dict, metrics: dict) -> str:
    entry = (verdicts.get(name) or {}).get(check)
    if not isinstance(entry, dict):
        return r"\NAMark"
    if not _check_informative(name, check, verdicts, metrics):
        return r"\NAMark"
    return r"\PassMark" if entry.get("passed") else r"\FailMark"


def _counts(name: str, verdicts: dict) -> tuple[int, int]:
    verdict = verdicts.get(name) or {}
    checks = [c for c in verdict.values() if isinstance(c, dict)]
    passed = verdict.get("passed_count")
    if passed is None:
        passed = sum(1 for c in checks if c.get("passed") and c.get("informative", True))
    informative = verdict.get("informative_count")
    if informative is None:
        informative = sum(1 for c in checks if c.get("informative", True))
    return int(passed), int(informative)


def _data_tex(payload: dict, cfg, ctx_info: dict, out_dir: str) -> str:
    metrics = payload.get("metrics") or {}
    config = payload.get("config") or {}
    histories = payload.get("histories") or {}
    verdicts = payload.get("verdicts") or {}
    lines: list[str] = []

    def macro(name: str, value) -> None:
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    def newif(name: str, value: bool) -> None:
        lines.append(rf"\newif\if{name}")
        lines.append(rf"\{name}{'true' if value else 'false'}")

    def mval(name: str, key: str, default: float = 0.0) -> float:
        return float((metrics.get(name) or {}).get(key, default))

    def cget(key: str, default=None):
        if key in config:
            return config[key]
        return getattr(cfg, key, default)

    for name, label in MODEL_MACROS.items():
        macro(f"TestAcc{label}", _pct(mval(name, "test_acc")))
        macro(f"Asr{label}", _pct(mval(name, "asr")))
        macro(f"ForgetAcc{label}", _pct(mval(name, "forget_acc")))
        macro(f"RetainAcc{label}", _pct(mval(name, "retain_acc")))
        macro(f"Mia{label}", f"{mval(name, 'mia_auc'):.3f}")
        macro(f"Attn{label}", f"{mval(name, 'trigger_attention'):.3f}")

        seconds = mval(name, "cost_seconds")
        macro(f"CostSec{label}", _seconds(seconds))
        macro(f"CostMin{label}", _minutes(seconds))
        macro(f"ClientRounds{label}", str(int(mval(name, "cost_rounds"))))

    macro(
        "AttnBaseline",
        f"{mval('original', 'trigger_attention_baseline', 0.0625):.3f}",
    )

    retrain_seconds = mval("retrain", "cost_seconds")

    def speedup(name: str) -> str:
        seconds = mval(name, "cost_seconds")
        return f"{retrain_seconds / seconds:.1f}" if seconds > 0 else "0.0"

    macro("SpeedupPga", speedup("pga"))
    macro("SpeedupFinetune", speedup("finetune"))
    macro("AsrDropPga", f"{(mval('original', 'asr') - mval('pga', 'asr')) * 100:.1f}")
    macro(
        "AsrGapPga",
        f"{(mval('pga', 'asr') - mval('retrain', 'asr')) * 100:.1f}",
    )

    num_clients = int(cget("num_clients", 10))
    macro("NumClients", str(num_clients))
    macro("NumRetained", str(num_clients - 1))
    macro("NumRounds", str(int(cget("rounds", 0))))
    macro("RecoveryRounds", str(int(cget("recovery_rounds", 0))))
    macro("PoisonPct", str(int(round(float(cget("poison_frac", 0.0)) * 100))))
    macro("Seed", str(int(cget("seed", 0))))
    macro("DeviceName", _esc(cget("device", "")))
    macro("BatchSize", str(int(cget("batch_size", 0))))
    macro("LocalEpochs", str(int(cget("local_epochs", 1))))
    macro("BaseLr", _plain_float(cget("lr", 0.0)))
    macro("FinetuneLr", _plain_float(cget("finetune_lr", 0.0)))
    macro("PgaLr", _plain_float(cget("pga_lr", 0.0)))
    macro("PgaRadiusFrac", f"{float(cget('pga_radius_frac', 0.0)):.2f}")
    macro("PgaClip", _plain_float(cget("pga_clip", 0.0)))
    macro("PgaTargetAccPct", f"{float(cget('pga_target_acc', 0.0)) * 100:.0f}")
    macro("Partition", _partition(config))
    macro("TrainSize", f"{int(ctx_info.get("train_size", 0)):,}".replace(",", "{,}"))
    macro("TestSize", f"{int(ctx_info.get("test_size", 0)):,}".replace(",", "{,}"))
    macro("TriggeredTestSize", f"{int(ctx_info.get("triggered_test_size", 0)):,}".replace(",", "{,}"))
    macro("ForgetSetSize", f"{int(ctx_info.get("forget_set_size", 0)):,}".replace(",", "{,}"))
    macro("PoisonedCount", f"{int(ctx_info.get("poisoned_count", 0)):,}".replace(",", "{,}"))
    macro("CamSamples", str(int(cget("cam_samples", 0))))
    model_params = _model_params(ctx_info)
    macro("ModelParams", f"{model_params:,}".replace(",", "{,}"))
    macro("RunDate", _esc(_run_date(out_dir, ctx_info)))

    pga_history = histories.get("pga") or []
    ascent = next(
        (h for h in pga_history if h.get("phase") == "ascent"),
        pga_history[0] if pga_history else {},
    )
    macro("PgaEpochsUsed", str(int(ascent.get("pga_epochs", cget("pga_epochs", 0)))))
    macro("PgaAscentTestAcc", _pct(ascent.get("test_acc", 0.0)))
    macro("PgaAscentAsr", _pct(ascent.get("asr", 0.0)))
    macro("PgaAscentForgetAcc", _pct(ascent.get("forget_acc", 0.0)))
    macro("PgaAscentSec", _seconds(ascent.get("seconds", 0.0)))

    for verdict_label, name in VERDICT_MODELS.items():
        for check_label, check in VERDICT_CHECKS.items():
            macro(
                f"Verdict{verdict_label}{check_label}",
                _verdict_mark(name, check, verdicts, metrics),
            )
        passed, _ = _counts(name, verdicts)
        macro(f"ChecksPassed{verdict_label}", str(passed))

    _, informative_count = _counts(
        "pga" if "pga" in verdicts else "finetune", verdicts
    )
    macro("ChecksInformative", str(informative_count))

    newif(
        "MiaInformative",
        _check_informative("pga", "mia_auc", verdicts, metrics),
    )
    newif(
        "PgaAllPassed",
        bool((verdicts.get("pga") or {}).get("all_passed", False)),
    )
    newif(
        "FinetuneAllPassed",
        bool((verdicts.get("finetune") or {}).get("all_passed", False)),
    )

    hero = ctx_info.get("hero") or {}
    macro("HeroTrue", _esc(hero.get("true", "")))
    hero_models = hero.get("models") or {}
    for name, label in MODEL_MACROS.items():
        entry = hero_models.get(name) or {}
        macro(f"HeroPred{label}", _esc(entry.get("pred", "")))
        confidence = entry.get("confidence")
        value = "" if confidence is None else str(int(round(float(confidence) * 100)))
        macro(f"HeroConf{label}", value)

    return "\n".join(lines) + "\n"


def _history_rows(histories: dict) -> str:
    original = {h["round"]: h for h in (histories.get("original") or [])}
    retrain = {h["round"]: h for h in (histories.get("retrain") or [])}
    rows = []
    for round_no in sorted(original):
        entry = original[round_no]
        ref = retrain.get(round_no)
        tail = (
            f"{_pct(ref['test_acc'])} & {_pct(ref['asr'])}"
            if ref is not None
            else "-- & --"
        )
        rows.append(
            f"{round_no} & {_pct(entry['test_acc'])} & {_pct(entry['asr'])} & "
            f"{tail} " + r"\\"
        )
    return "\n".join(rows) + ("\n" if rows else "")


def _recovery_rows(histories: dict) -> str:
    pga_history = histories.get("pga") or []
    finetune = {h["round"]: h for h in (histories.get("finetune") or [])}
    ascent = next(
        (h for h in pga_history if h.get("phase") == "ascent"),
        pga_history[0] if pga_history else None,
    )
    recovery = sorted(
        (h for h in pga_history if h.get("phase") == "recovery"),
        key=lambda h: h["round"],
    )
    rows = []
    base_round = None
    if ascent is not None:
        base_round = int(ascent["round"])
        rows.append(
            f"Ascent & -- & -- & {_pct(ascent['test_acc'])} & "
            f"{_pct(ascent['asr'])} " + r"\\"
        )
    for entry in recovery:
        offset = int(entry["round"]) - base_round if base_round is not None else 0
        label = f"T+{offset}"
        ft = finetune.get(entry["round"])
        ft_acc = _pct(ft["test_acc"]) if ft is not None else "--"
        ft_asr = _pct(ft["asr"]) if ft is not None else "--"
        rows.append(
            f"{label} & {ft_acc} & {ft_asr} & {_pct(entry['test_acc'])} & "
            f"{_pct(entry['asr'])} " + r"\\"
        )
    return "\n".join(rows) + ("\n" if rows else "")


def _read_group(text: str, start: int) -> tuple[str, int]:
    """Return the brace group starting at ``text[start] == '{'`` and the index after it."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    raise ValueError("unbalanced braces in index entry")


def _page_key(page: str):
    return (0, int(page)) if page.isdigit() else (-1, page)


def build_index(idx_path: str, ind_path: str) -> int:
    """Turn ``main.idx`` into a typeset ``main.ind`` (replaces makeindex).

    Supports ``main!sub`` entries and ``|see{target}``; emits the macros
    ``\\ixletter``, ``\\ixitem``, ``\\ixhead``, ``\\ixsub`` and ``\\ixsee``
    that the template defines. Returns the number of main entries.
    """
    entries: dict[str, dict] = {}
    if not os.path.isfile(idx_path):
        return 0
    with open(idx_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("\\indexentry"):
                continue
            key, pos = _read_group(line, line.index("{"))
            page, _ = _read_group(line, line.index("{", pos))
            term, _, fmt = key.partition("|")
            see = re.search(r"see\s*\{([^}]*)\}", fmt)
            main, _, sub = term.partition("!")
            main, sub = main.strip(), sub.strip()
            node = entries.setdefault(main, {"pages": set(), "subs": {}, "see": None})
            if see:
                node["see"] = see.group(1).strip()
            elif sub:
                node["subs"].setdefault(sub, set()).add(page)
            else:
                node["pages"].add(page)

    def pages(values: set) -> str:
        ordered = sorted(values, key=_page_key)
        return ", ".join(f"\\hyperpage{{{p}}}" for p in ordered)

    lines: list[str] = []
    letter = None
    for main in sorted(entries, key=lambda t: t.lower()):
        node = entries[main]
        first = main[0].upper()
        if first != letter:
            letter = first
            lines.append(f"\\ixletter{{{letter}}}")
        if node["see"] and not node["pages"] and not node["subs"]:
            lines.append(f"\\ixsee{{{main}}}{{{node['see']}}}")
            continue
        if node["pages"]:
            lines.append(f"\\ixitem{{{main}}}{{{pages(node['pages'])}}}")
        else:
            lines.append(f"\\ixhead{{{main}}}")
        for sub in sorted(node["subs"], key=str.lower):
            lines.append(f"\\ixsub{{{sub}}}{{{pages(node['subs'][sub])}}}")
    with open(ind_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(entries)


def _engine_command() -> list[str] | None:
    """Prefer a local Tectonic; otherwise latexmk with LuaLaTeX."""
    tectonic = shutil.which("tectonic")
    if tectonic:
        command = [tectonic]
        if os.environ.get("FEDUNLEARN_TEX_OFFLINE") == "1":
            # use only the files already in Tectonic's cache; never touch the network
            command.append("--only-cached")
        return command + ["--keep-intermediates", "--keep-logs", "main.tex"]
    latexmk = shutil.which("latexmk")
    if latexmk:
        return [
            latexmk,
            "-lualatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "main.tex",
        ]
    return None


def _run_engine(command: list[str], report_dir: str) -> None:
    try:
        result = subprocess.run(
            command, cwd=report_dir, capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        print("The LaTeX build timed out after 900 seconds.")
        raise SystemExit(1)
    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).splitlines()
        print("LaTeX build failed:", " ".join(command))
        print("\n".join(output[-40:]))
        raise SystemExit(1)


def _compile(report_dir: str, out_dir: str) -> str | None:
    command = _engine_command()
    if command is None:
        print("No LaTeX engine found (tectonic or latexmk); the sources are ready in")
        print(f"  {report_dir}")
        print("Install Tectonic and rerun with --stage report.")
        return None
    # pass 1 writes main.idx, the index is built from it, pass 2 typesets it
    _run_engine(command, report_dir)
    count = build_index(
        os.path.join(report_dir, "main.idx"), os.path.join(report_dir, "main.ind")
    )
    _run_engine(command, report_dir)
    print(f"[report] index: {count} entries")
    pdf_source = os.path.join(report_dir, "main.pdf")
    pdf_target = os.path.join(out_dir, "report.pdf")
    shutil.copy2(pdf_source, pdf_target)
    return pdf_target


def build(
    payload: dict,
    cfg,
    ctx_info: dict,
    out_dir: str,
    template_dir: str = "report_template",
    compile_pdf: bool = True,
) -> dict:
    report_dir = os.path.join(out_dir, "report")
    generated_dir = os.path.join(report_dir, "generated")
    template_path = os.path.join(template_dir, "main.tex")
    if not os.path.isfile(template_path):
        raise SystemExit(
            f"LaTeX template not found at {template_path!r}; expected a hand-written "
            "main.tex in the template directory."
        )

    os.makedirs(generated_dir, exist_ok=True)
    shutil.copy2(template_path, os.path.join(report_dir, "main.tex"))

    source_figures = os.path.join(out_dir, "figures")
    target_figures = os.path.join(report_dir, "figures")
    os.makedirs(target_figures, exist_ok=True)
    if os.path.isdir(source_figures):
        for name in sorted(os.listdir(source_figures)):
            source = os.path.join(source_figures, name)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(target_figures, name))

    data_path = os.path.join(generated_dir, "data.tex")
    history_path = os.path.join(generated_dir, "history_rows.tex")
    recovery_path = os.path.join(generated_dir, "recovery_rows.tex")
    with open(data_path, "w", encoding="utf-8") as handle:
        handle.write(_data_tex(payload, cfg, ctx_info, out_dir))
    with open(history_path, "w", encoding="utf-8") as handle:
        handle.write(_history_rows(payload.get("histories") or {}))
    with open(recovery_path, "w", encoding="utf-8") as handle:
        handle.write(_recovery_rows(payload.get("histories") or {}))

    pdf_path = _compile(report_dir, out_dir) if compile_pdf else None
    return {
        "report_dir": report_dir,
        "data_tex": data_path,
        "history_rows": history_path,
        "recovery_rows": recovery_path,
        "pdf": pdf_path,
    }
