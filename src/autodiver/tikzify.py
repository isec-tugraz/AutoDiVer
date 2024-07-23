#!/usr/bin/env python3
"""
create tikz code and compile to pdf for differential charactersitics
"""
from __future__ import annotations

import argparse
from pathlib import Path
from shutil import which
import sys
import subprocess as sp

from .cipher_model import DifferentialCharacteristic
from .gift.gift_model import Gift64Characteristic, Gift128Characteristic


def main():
    ciphers: dict[str, type[DifferentialCharacteristic]] = {
        "gift64": Gift64Characteristic,
        "gift128": Gift128Characteristic,
    }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cipher', choices=ciphers.keys())
    # parser.add_argument('-o', '--output', default=None, help="output .tex file")
    parser.add_argument('-nc', '--no-compile', dest="compile", action="store_false", help="do not compile the .tex file")
    parser.add_argument('-v', '--verbose', action="store_true", help="show LaTeX compile log")
    parser.add_argument('characteristic', type=Path, help="file containing the differential characteristic")

    args = parser.parse_args()

    CharacteristicType = ciphers[args.cipher]
    char = CharacteristicType.load(args.characteristic)
    tex_file = args.characteristic.with_suffix(".tex")
    tex_file.write_text(char.tikzify())

    if args.compile:
        latexmk = which("latexmk")
        if latexmk is None:
            print("latexmk not found, skipping compilation", file=sys.stderr)
            return 1

        output = None if args.verbose else sp.DEVNULL
        try:
            sp.check_call([latexmk, "-pdf", tex_file], stdout=output, stderr=output)
            sp.check_call([latexmk, "-c", tex_file], stdout=output, stderr=output)
        except sp.CalledProcessError as e:
            print(f"latexmk failed with exit code {e.returncode}", file=sys.stderr)
            return e.returncode


if __name__ == "__main__":
    raise SystemExit(main())
