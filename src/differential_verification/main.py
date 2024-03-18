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
from .cipher_model import CountResult, SboxCipher, DifferentialCharacteristic
from .gift64.gift_model import Gift64
from .midori64.midori64_model import Midori64
from .midori128.midori128_model import Midori128
from .ascon.ascon_model import Ascon, AsconCharacteristic
from .skinny.skinny_model import Skinny128, Skinny64, Skinny128Characteristic, Skinny64Characteristic
log = logging.getLogger(__name__)
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
        "midori64": (Midori64, DifferentialCharacteristic),
        "midori128": (Midori128, DifferentialCharacteristic),
        "skinny128": (Skinny128, Skinny128Characteristic),
        "skinny64": (Skinny64, Skinny64Characteristic),
        "ascon": (Ascon, AsconCharacteristic),
    }
    commands: dict[str, Callable[[SboxCipher, argparse.Namespace], None|CountResult]] = {
        'count-tweaks': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=False, count_tweak=True),
        'count-keys': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=True, count_tweak=False),
        'count-tweakeys': lambda cipher, args: cipher.count_tweakey_space(args.epsilon, args.delta, count_key=True, count_tweak=True),
        'count-tweaks-sat': lambda cipher, args: cipher.count_tweakey_space_sat_solver(1_000, count_key=False, count_tweak=True),
        'count-keys-sat': lambda cipher, args: cipher.count_tweakey_space_sat_solver(1_000, count_key=True, count_tweak=False),
        'count-tweakeys-sat': lambda cipher, args: cipher.count_tweakey_space_sat_solver(1_000, count_key=True, count_tweak=True),
        'count-prob': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta),
        'count-prob-fixed-key': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_key=True),
        'count-prob-fixed-tweak': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_tweak=True),
        'count-prob-fixed-tweakey': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_tweak=True, fixed_key=True),
        'count-prob-fixed-pt': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_pt=True),
        'count-prob-fixed-pt-and-tweak': lambda cipher, args: cipher.count_probability(args.epsilon, args.delta, fixed_pt=True, fixed_tweak=True),
        'solve': lambda cipher, _args: cipher.solve(),
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
    git_cmd = shutil.which('git')
    git_commit = git_cmd and sp.check_output([git_cmd, 'rev-parse', 'HEAD']).decode().strip()
    git_changed_files = git_cmd and sp.check_output([git_cmd, 'status', '--porcelain', '-uno', '-z']).decode().strip('\0').split('\0')
    log.info(f"version: {version}, git_commit: {git_commit}, git_changed_files: {git_changed_files}")
    log.info("arguments: %s", vars(args), extra={"cli_args": vars(args), "git_commit": git_commit, "git_changed_files": git_changed_files, "version": version})
    log.info(f"reading trail from {args.trail!r}")
    Cipher, Characteristic = ciphers[args.cipher]
    char = Characteristic.load(args.trail)
    log.info(f"loaded characteristic with {char.num_rounds} rounds from {args.trail!r}")
    cipher = Cipher(char)
    ddt_prob = char.log2_ddt_probability(Cipher.ddt)
    log.info(f"ddt probability: 2**{ddt_prob:.1f}")
    ddt_prob_1_plus = np.log2(cipher.ddt[char.sbox_in[1:], char.sbox_out[1:]] / len(cipher.ddt)).sum()
    log.info(f"ddt probability r1+: 2**{ddt_prob_1_plus:.1f}")
    log.info(f"generated {cipher.cnf!r}")
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