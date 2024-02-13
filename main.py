#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging.config
from pathlib import Path
import subprocess as sp
import sys
from typing import Optional, Callable
import numpy as np
from IPython import start_ipython
from cipher_model import CountResult, SboxCipher, DifferentialCharacteristic
from gift64.gift64 import Gift64
from skinny.skinny128 import Skinny128, SkinnyCharacteristic
log = logging.getLogger('main')
def setup_logging(filename: Optional[Path] = None):
    config_file = Path(__file__).parent / 'log_config.json'
    with config_file.open('r') as f:
        config = json.load(f)
    if filename:
        config['handlers']['file']['filename'] = filename
    logging.getLogger().setLevel(logging.DEBUG)
    logging.config.dictConfig(config)
def main():
    ciphers: dict[str, tuple[type[SboxCipher], type[DifferentialCharacteristic]]] = {
        "gift64": (Gift64, DifferentialCharacteristic),
        "skinny128": (Skinny128, SkinnyCharacteristic),
    }
    commands: dict[str, Callable[[SboxCipher, argparse.Namespace], None|CountResult]] = {
        'count-tweaks': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=False, count_tweak=True),
        'count-keys': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=True, count_tweak=False),
        'count-tweakeys': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=True, count_tweak=True),
        'count-prob': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta),
        'count-prob-fixed-key': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta),
        'count-prob-fixed-tweak': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_key=True),
        'count-prob-fixed-tweakey': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_tweak=True, fixed_key=True),
        'count-prob-fixed-pt': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_pt=True),
        'count-prob-fixed-pt-and-tweak': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_pt=True, fixed_tweak=True),
        'embed': lambda _cipher, _args: __import__('IPython').embed(),
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cipher', choices=ciphers.keys())
    parser.add_argument('trail', type=Path, help='Text file containing the sbox input and output differences.\n'\
                                      'Input and output differences are listed on separate lines.')
    parser.add_argument('--epsilon', type=float, default=0.8)
    parser.add_argument('--delta', type=float, default=0.2)
    parser.add_argument('--cnf', type=str, help="file to save CNF in DIMACS format")
    parser.add_argument('commands', choices=commands.keys(), nargs='+', help="commands to execute")
    args = parser.parse_args()
    setup_logging(args.trail.with_suffix('.jsonl'))
    git_commit = sp.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    git_changed_files = sp.check_output(['git', 'status', '--porcelain', '-uno', '-z']).decode().strip('\0').split('\0')
    log.info("arguments: %s", vars(args), extra={"cli_args": vars(args), "git_commit": git_commit, "git_changed_files": git_changed_files})
    log.info(f"reading trail from {args.trail!r}")
    Cipher, Characteristic = ciphers[args.cipher]
    char = Characteristic.load(args.trail)
    cipher = Cipher(char)
    ddt_prob = char.log2_ddt_probability(Cipher.ddt)
    log.info(f"ddt probability: 2**{ddt_prob:.1f}")
    ddt_prob_1_plus = np.log2(cipher.ddt[char.sbox_in[1:], char.sbox_out[1:]] / len(cipher.ddt)).sum()
    log.info(f"ddt probability r1+: 2**{ddt_prob_1_plus:.1f}")
    if args.cnf:
        with open(args.cnf, 'w') as f:
            f.write(cipher.cnf.to_dimacs())
        log.info(f"wrote {cipher.cnf!r} to {args.cnf}")
    count_results = []
    for command in args.commands:
        if command == 'embed':
            sys.argv = sys.argv[:1]
            start_ipython(user_ns=globals()|locals())
            continue
        res = commands[command](cipher, args)
        if res is not None:
            count_results.append(res)
    # from IPython import embed; embed()
if __name__ == "__main__":
    raise SystemExit(main())