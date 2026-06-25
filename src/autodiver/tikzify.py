#!/usr/bin/env python3
"""
create tikz code and compile to pdf for differential characteristics
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import tempfile
from pathlib import Path
from shutil import which
import sys
import subprocess as sp

from .cipher_model import DifferentialCharacteristic


def latex_support_dir() -> Path:
    """Locate the directory holding the cipher .sty files and tikz library."""
    # the latex/ directory sits at the repository root, next to src/
    candidate = Path(__file__).resolve().parents[2] / "latex"
    if candidate.is_dir():
        return candidate
    # fall back to a latex/ directory in the current working directory
    return Path.cwd() / "latex"


def main():
    ciphers: dict[str, tuple[str, str, str, bool]] = {
        "ascon": ("autodiver.ascon.ascon_model", "Ascon", "AsconCharacteristic", False),
        "gift64": ("autodiver.gift.gift_model", "Gift64", "Gift64Characteristic", True),
        "gift128": ("autodiver.gift.gift_model", "Gift128", "Gift128Characteristic", True),
        "midori64": ("autodiver.midori64.midori64_model", "Midori64", "Midori64Characteristic", False),
        "midori128": ("autodiver.midori128.midori128_model", "Midori128", "Midori128Characteristic", False),
        "present80": ("autodiver.present.present_model", "Present80", "PresentCharacteristic", True),
        "pyjamask96": ("autodiver.pyjamask.pyjamask96_model", "Pyjamask_with_Keyschedule", "Pyjamask96Characteristic",
                       False),
        "rectangle128": ("autodiver.rectangle128.rectangle_model", "Rectangle128", "RectangleCharacteristic", False),
        "skinny64": ("autodiver.skinny.skinny_model", "Skinny64", "Skinny64Characteristic", True),
        "skinny128": ("autodiver.skinny.skinny_model", "Skinny128", "Skinny128Characteristic", True),
        "speck32": ("autodiver.speck.speck_model", "Speck32LongKey", "Speck32Characteristic", True),
        "speck48": ("autodiver.speck.speck_model", "Speck48LongKey", "Speck48Characteristic", True),
        "speck64": ("autodiver.speck.speck_model", "Speck64LongKey", "Speck64Characteristic", True),
        "speck96": ("autodiver.speck.speck_model", "Speck96LongKey", "Speck96Characteristic", True),
        "speck128": ("autodiver.speck.speck_model", "Speck128LongKey", "Speck128Characteristic", True),
        "speedy192": ("autodiver.speedy192.speedy192_model", "Speedy192", "Speedy192Characteristic", False),
        "warp": ("autodiver.warp128.warp128_model", "WARP128", "WarpCharacteristic", True),
    }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cipher', choices=ciphers.keys())
    parser.add_argument('-nc', '--no-compile', dest="compile", action="store_false", help="do not compile the .tex file")
    parser.add_argument('-k', '--keep-tikz', action="store_true",
                        help="keep the generated .tex file next to the characteristic (and its pdf)")
    parser.add_argument('-v', '--verbose', action="store_true", help="show LaTeX compile log")
    parser.add_argument('characteristic', type=Path, help="file containing the differential characteristic")

    args = parser.parse_args()

    module_name, _cipher_type_name, characteristic_type_name, _ = ciphers[args.cipher]

    module = importlib.import_module(module_name)
    CharacteristicType: type[DifferentialCharacteristic] = getattr(module, characteristic_type_name)

    char = CharacteristicType.load(args.characteristic)
    tex = char.tikzify()

    char_path = args.characteristic.resolve()
    tex_dest = char_path.with_suffix(".tex")
    pdf_dest = char_path.with_suffix(".pdf")

    if not args.compile:
        tex_dest.write_text(tex)
        print(f"wrote {tex_dest}")
        return 0

    latexmk = which("latexmk")
    if latexmk is None:
        tex_dest.write_text(tex)
        print(f"latexmk not found, skipping compilation; wrote {tex_dest}", file=sys.stderr)
        return 1

    output = None if args.verbose else sp.DEVNULL

    # compile in a throw-away directory containing the cipher .sty files so the
    # working directory and the characteristic's directory stay clean
    with tempfile.TemporaryDirectory(prefix="autodiver-tikzify-") as tmpdir:
        tmp = Path(tmpdir)
        support_dir = latex_support_dir()
        for support_file in (*support_dir.glob("*.sty"), *support_dir.glob("*.code.tex")):
            shutil.copy(support_file, tmp)

        tex_file = tmp / tex_dest.name
        tex_file.write_text(tex)

        try:
            sp.check_call([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
                          cwd=tmp, stdout=output, stderr=output)
        except sp.CalledProcessError as e:
            print(f"latexmk failed with exit code {e.returncode}", file=sys.stderr)
            return e.returncode

        shutil.copy(tex_file.with_suffix(".pdf"), pdf_dest)
        print(f"wrote {pdf_dest}")
        if args.keep_tikz:
            shutil.copy(tex_file, tex_dest)
            print(f"wrote {tex_dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
