"""
Smoke test for ``autodiver.tikzify``: for every cipher that supports tikz
rendering, load one differential characteristic, render it to TikZ/LaTeX and
make sure ``latexmk`` compiles it to a PDF.

The whole module is skipped when ``latexmk`` is not installed.

Set the environment variable ``AUTODIVER_KEEP_TIKZ_OUTPUT`` to keep the
generated ``.tex``/``.pdf`` files for manual visual inspection instead of
compiling them in a throw-away temporary directory:

    AUTODIVER_KEEP_TIKZ_OUTPUT=1 pytest tests/test_tikzify.py
    AUTODIVER_KEEP_TIKZ_OUTPUT=/tmp/tikz pytest tests/test_tikzify.py

When set to a truthy value the output is kept under
``$TMPDIR/autodiver-tikzify/<cipher>/``; when set to a path that directory is
used as the base instead.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess as sp
import tempfile
from contextlib import contextmanager
from pathlib import Path
from shutil import which

import pytest

from autodiver.characteristic import DifferentialCharacteristic
from autodiver.tikzify import CIPHERS

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEX_DIR = REPO_ROOT / "latex"
TRAILS = REPO_ROOT / "trails"

# one characteristic per tikzify-capable cipher
CHARACTERISTICS: dict[str, Path] = {
    "gift64": TRAILS / "gift64" / "gift64_toy_char.txt",
    "gift128": TRAILS / "gift128" / "gift128_r12_LLL+_table_5.txt",
    "present80": TRAILS / "present" / "present_toy.npz",
    "skinny64": TRAILS / "skinny64" / "skinny64_tk3_ddh+21_r15_table_9.npz",
    "skinny128": TRAILS / "skinny128" / "skinny128_tk3_NPE23_r10_figure_8.npz",
    "speck32": TRAILS / "speck" / "speck32_r9_ALLW14_table_7.npz",
    "speck48": TRAILS / "speck" / "speck48_r10_ALLW14_table_7_fixed.npz",
    "speck64": TRAILS / "speck" / "speck64_r13_ALLW14_table_9.npz",
    "speck96": TRAILS / "speck" / "speck96_r15_BR22_table_22c.npz",
    "speck128": TRAILS / "speck" / "speck128_r20_BR22_table_23a.npz",
    "warp": TRAILS / "warp" / "warp_KY22_r18_table_8.npz",
}

SUPPORTED_CIPHERS = sorted(name for name, (*_, tikz) in CIPHERS.items() if tikz)

pytestmark = pytest.mark.skipif(which("latexmk") is None, reason="latexmk is not installed")


def test_all_supported_ciphers_have_a_characteristic():
    """Guard: adding a new tikzify-capable cipher must add a test characteristic."""
    assert sorted(CHARACTERISTICS) == SUPPORTED_CIPHERS


@contextmanager
def work_dir(cipher: str):
    """Yield a directory prepared for latexmk (the .sty files copied in).

    Cleaned up afterwards unless AUTODIVER_KEEP_TIKZ_OUTPUT is set, in which
    case it is kept and its location is reported.
    """
    keep = os.environ.get("AUTODIVER_KEEP_TIKZ_OUTPUT")
    if keep:
        base = Path(keep) if keep not in ("1", "true", "yes", "on") else Path(tempfile.gettempdir()) / "autodiver-tikzify"
        path = base / cipher
        path.mkdir(parents=True, exist_ok=True)
    else:
        path = Path(tempfile.mkdtemp(prefix=f"tikzify-{cipher}-"))

    # latexmk needs the cipher .sty files and the tikz library next to the .tex
    for sty in [*LATEX_DIR.glob("*.sty"), *LATEX_DIR.glob("*.code.tex")]:
        shutil.copy(sty, path)

    try:
        yield path
    finally:
        if keep:
            print(f"\nkept tikzify output for {cipher} in {path}")
        else:
            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize("cipher", SUPPORTED_CIPHERS)
def test_tikzify_compiles(cipher: str):
    module_name, _cipher_cls, char_cls_name, _ = CIPHERS[cipher]
    char_file = CHARACTERISTICS[cipher]
    assert char_file.is_file(), f"missing characteristic file {char_file}"

    module = importlib.import_module(module_name)
    CharacteristicType: type[DifferentialCharacteristic] = getattr(module, char_cls_name)

    char = CharacteristicType.load(char_file)
    tex = char.tikzify()
    assert tex.strip(), "tikzify() produced empty output"

    latexmk = which("latexmk")
    assert latexmk is not None  # guarded by pytestmark

    with work_dir(cipher) as path:
        tex_file = path / f"{cipher}.tex"
        tex_file.write_text(tex)

        proc = sp.run(
            [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
            cwd=path, capture_output=True, text=True,
        )
        pdf_file = tex_file.with_suffix(".pdf")
        assert proc.returncode == 0 and pdf_file.is_file(), (
            f"latexmk failed for {cipher} (exit {proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr ---\n{proc.stderr[-1000:]}"
        )
