#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging.config
from pathlib import Path
import shutil
import subprocess as sp
import sys
from typing import Optional, Callable
import numpy as np
from IPython import start_ipython

from .import version
from .cipher_model import CountResult, SboxCipher, DifferentialCharacteristic, UnsatException
from .gift64.gift_model import Gift64
from .gift128.gift_model import Gift128
from .rectangle128.rectangle_model import Rectangle128, RectangleLongKey
from .midori64.midori64_model import Midori64, Midori64Characteristic
from .midori128.midori128_model import Midori128, Midori128Characteristic
from .warp128.warp128_model import WARP128
from .speedy192.speedy192_model import Speedy192, Speedy192Characteristic
from .ascon.ascon_model import Ascon, AsconCharacteristic
from .skinny.skinny_model import Skinny128, Skinny64, Skinny128Characteristic, Skinny64Characteristic
from .present.present_model import Present80, PresentLongKey, PresentCharacteristic

log = logging.getLogger(__name__)
def setup_logging(filename: Optional[Path] = None):
    config_file = Path(__file__).parent / 'log_config.json'
    with config_file.open('r') as f:
        config = json.load(f)
    if filename:
        config['handlers']['file']['filename'] = filename
    logging.getLogger().setLevel(logging.DEBUG)
    logging.config.dictConfig(config)
def parse_slice(s: str) -> slice:
    start, sep, stop = s.partition('..')
    if sep == '':
        raise argparse.ArgumentTypeError(f"invalid slice {s!r} -> expected 'start..stop'")
    start = int(start) if start else None
    stop = int(stop) if stop else None
    return slice(start, stop)

def FilePath(path: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"{path!r} is not a file")
    return p

def solve_cipher_interactive(cipher: SboxCipher):
    try:
        cipher.solve()
    except UnsatException:
        pass

def main():
    ciphers: dict[str, tuple[type[SboxCipher], type[DifferentialCharacteristic]]] = {
        "warp": (WARP128, DifferentialCharacteristic),
        "speedy192": (Speedy192, Speedy192Characteristic),
        "gift64": (Gift64, DifferentialCharacteristic),
        "gift128": (Gift128, DifferentialCharacteristic),
        "present80": (Present80, PresentCharacteristic),
        "present-long-key": (PresentLongKey, PresentCharacteristic),
        "midori64": (Midori64, Midori64Characteristic),
        "midori128": (Midori128, Midori128Characteristic),
        "skinny128": (Skinny128, Skinny128Characteristic),
        "skinny64": (Skinny64, Skinny64Characteristic),
        "ascon": (Ascon, AsconCharacteristic),
        "rectangle128": (Rectangle128, DifferentialCharacteristic),
        "rectangle-long-key": (RectangleLongKey, DifferentialCharacteristic),
    }
    commands: dict[str, Callable[[SboxCipher, argparse.Namespace], None|CountResult]] = {
        'count-tweaks': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=False, count_tweak=True),
        'count-keys': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=True, count_tweak=False),
        'count-tweakeys': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=True, count_tweak=True),
        'count-tweaks-sat': lambda cipher, args: cipher.count_tweakey_space_sat_solver(1_000, count_key=False, count_tweak=True),
        'count-keys-sat': lambda cipher, args: cipher.count_tweakey_space_sat_solver(1_000, count_key=True, count_tweak=False),
        'count-tweakeys-sat': lambda cipher, args: cipher.count_tweakey_space_sat_solver(1_000, count_key=True, count_tweak=True),
        'count-tweaks-lin': lambda cipher, args: cipher.count_lin_tweakey_space(count_key=False, count_tweak=True),
        'count-keys-lin': lambda cipher, args: cipher.count_lin_tweakey_space(count_key=True, count_tweak=False),
        'count-tweakeys-lin': lambda cipher, args: cipher.count_lin_tweakey_space(count_key=True, count_tweak=True),
        'count-prob': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta),
        'count-prob-fixed-key': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_key=True),
        'count-prob-fixed-tweak': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_tweak=True),
        'count-prob-fixed-tweakey': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_tweak=True, fixed_key=True),
        'count-prob-fixed-pt': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_pt=True),
        'count-prob-fixed-pt-and-tweak': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_pt=True, fixed_tweak=True),
        'solve': lambda cipher, _args: solve_cipher_interactive(cipher),
        'find-conflict': lambda cipher, _args: cipher.find_conflict(),
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cipher', choices=ciphers.keys())
    parser.add_argument('trail', type=FilePath, help='Text/Numpy file containing the sbox input and output differences.\n'\
                                      'Input and output differences are listed on separate lines.')
    parser.add_argument('--epsilon', type=float, default=0.8)
    parser.add_argument('--delta', type=float, default=0.2)
    parser.add_argument('--cnf', type=str, help="file to save CNF in DIMACS format")
    parser.add_argument('--sbox-assumptions', action='store_true', help="add assumption variables for all S-boxes")
    parser.add_argument('--embed', action='store_true', help="launch IPython shell after executing command")
    parser.add_argument('commands', choices=commands.keys(), nargs=1, help="command to execute")
    args = parser.parse_args()
    setup_logging(args.trail.with_suffix('.jsonl'))

    if 'find-conflict' in args.commands and not args.sbox_assumptions:
        log.warning("command 'find-conflict' requires --sbox-assumptions, adding it automatically")
        args.sbox_assumptions = True

    git_cmd = shutil.which('git')
    git_commit = git_cmd and sp.check_output([git_cmd, 'rev-parse', 'HEAD']).decode().strip()
    git_changed_files = git_cmd and sp.check_output([git_cmd, 'status', '--porcelain', '-uno', '-z']).decode().strip('\0').split('\0')
    log.info(f"version: {version}, git_commit: {git_commit}, git_changed_files: {git_changed_files}")
    log.info("arguments: %s", vars(args), extra={"cli_args": vars(args), "git_commit": git_commit, "git_changed_files": git_changed_files, "version": version})
    Cipher, Characteristic = ciphers[args.cipher]
    try:
        char = Characteristic.load(args.trail)
        if char.file_path is None:
            log.warning(f"file path not stored in characteristic object")
            char.file_path = args.trail
    except OSError as e:
        log.error(e)
        return 1
    log.info(f"loaded characteristic with {char.num_rounds} rounds from {args.trail}")
    cipher = Cipher(char, model_sbox_assumptions=args.sbox_assumptions)
    ddt_prob = char.log2_ddt_probability(Cipher.ddt)
    log.info(f"ddt probability: 2**{ddt_prob:.1f}")
    log.info(f"generated {cipher.cnf!r}")
    if args.cnf:
        with open(args.cnf, 'w') as f:
            f.write(cipher.cnf.to_dimacs())
        log.info(f"wrote {cipher.cnf!r} to {args.cnf}")
    count_results = []
    for command in args.commands:
        res = commands[command](cipher, args)
        if res is not None:
            count_results.append(res)
    if args.embed:
        start_ipython(user_ns=globals()|locals())
if __name__ == "__main__":
    raise SystemExit(main())
