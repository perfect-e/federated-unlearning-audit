"""Compile the report once with placeholder data so Tectonic caches every
package and font the template needs. Used while building the Docker image, so
that later report builds run without network access.

    python scripts/warm_tex_cache.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("FEDUNLEARN_TEX_OFFLINE", None)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fedunlearn import latex_report  # noqa: E402
from fedunlearn.config import Config  # noqa: E402
from tests.test_latex_report import _ctx_info, _payload  # noqa: E402


def main() -> int:
    template_dir = os.path.join(ROOT, "report_template")
    with open(os.path.join(template_dir, "main.tex"), encoding="utf-8") as fh:
        names = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", fh.read())))
    with tempfile.TemporaryDirectory() as out_dir:
        figures = os.path.join(out_dir, "figures")
        os.makedirs(figures)
        for name in names:
            fig = plt.figure(figsize=(2, 2))
            fig.savefig(os.path.join(figures, name))
            plt.close(fig)
        result = latex_report.build(
            _payload(), Config(), _ctx_info(), out_dir, template_dir=template_dir
        )
        if not result.get("pdf"):
            print("warm-up build produced no PDF")
            return 1
    print(f"TeX cache warmed ({len(names)} placeholder figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
