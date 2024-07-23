#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging.config
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess as sp
import sys
from typing import Optional, Callable, Any

import numpy as np
from IPython import start_ipython
import click

from .import version
from .cipher_model import CountResult, SboxCipher, DifferentialCharacteristic, UnsatException
from .gift.gift_model import Gift64, Gift64Characteristic, Gift128, Gift128Characteristic
from .rectangle128.rectangle_model import Rectangle128, RectangleLongKey
from .midori64.midori64_model import Midori64, Midori64Characteristic
from .midori128.midori128_model import Midori128, Midori128Characteristic
from .warp128.warp128_model import WARP128
from .speedy192.speedy192_model import Speedy192, Speedy192Characteristic
from .ascon.ascon_model import Ascon, AsconCharacteristic
from .skinny.skinny_model import Skinny128, Skinny64, Skinny128Characteristic, Skinny64Characteristic
from .present.present_model import Present80, PresentLongKey, PresentCharacteristic

log = logging.getLogger(__name__)


@dataclass
class GlobalArgs:
    characteristic: DifferentialCharacteristic
    cipher: SboxCipher

def setup_logging(filename: Optional[Path] = None):
    config_file = Path(__file__).parent / 'log_config.json'
    with config_file.open('r') as f:
        config = json.load(f)

    if filename:
        config['handlers']['file']['filename'] = filename

    logging.getLogger().setLevel(logging.DEBUG)
    logging.config.dictConfig(config)


_ciphers: dict[str, tuple[type[SboxCipher], type[DifferentialCharacteristic]]] = {
    "warp": (WARP128, DifferentialCharacteristic),
    "speedy192": (Speedy192, Speedy192Characteristic),
    "gift64": (Gift64, Gift64Characteristic),
    "gift128": (Gift128, Gift128Characteristic),
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


@click.group()
@click.argument('cipher_name', type=click.Choice(list(_ciphers.keys())), required=True)
@click.argument('characteristic_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True)
@click.option('--sbox-assumptions', is_flag=True, help="add assumption variables for all S-boxes")
@click.pass_context
def cli(ctx, cipher_name: str, characteristic_path: str|Path, sbox_assumptions: bool) -> None:
    characteristic_path = Path(characteristic_path)
    setup_logging(characteristic_path.with_suffix('.jsonl'))
    git_cmd = shutil.which('git')
    git_commit = git_cmd and sp.check_output([git_cmd, 'rev-parse', 'HEAD']).decode().strip()
    git_changed_files = git_cmd and sp.check_output([git_cmd, 'status', '--porcelain', '-uno', '-z']).decode().strip('\0').split('\0')
    log.info(f"version: {version}, git_commit: {git_commit}, git_changed_files: {git_changed_files}")
    log.debug("arguments: %s", sys.argv, extra={"cli_args": sys.argv, "git_commit": git_commit, "git_changed_files": git_changed_files, "version": version})

    Cipher, Characteristic = _ciphers[cipher_name]
    characteristic = Characteristic.load(characteristic_path)
    cipher = Cipher(characteristic, model_sbox_assumptions=sbox_assumptions)
    ddt_prob_log2 = characteristic.log2_ddt_probability(Cipher.ddt)
    log.info(f"loaded characteristic with {characteristic.num_rounds} rounds from {characteristic_path} with ddt probability 2**{ddt_prob_log2:.1f}")

    if characteristic.file_path is None:
        log.warning(f"file path not stored in characteristic object")
        characteristic.file_path = characteristic_path

    log.info(f"generated {cipher.cnf!r}")
    ctx.obj = GlobalArgs(characteristic, cipher)

def ensure_executables(*executables: str) -> None:
    missing = [exe for exe in executables if not shutil.which(exe)]
    if missing:
        if len(missing) == 1:
            raise click.UsageError(f"missing executable in $PATH: {missing[0]}")
        raise click.UsageError(f"missing executables in $PATH: {', '.join(missing)}")

def ensure_cipher_comatible(cipher: SboxCipher, needs_key: bool, needs_tweak: bool) -> None:
    if cipher.key_size == 0 and needs_key:
        raise click.UsageError(f"{cipher.cipher_name} has no key")
    if cipher.tweak_size == 0 and needs_tweak:
        raise click.UsageError(f"{cipher.cipher_name} has no tweak")

@cli.command()
@click.option('--epsilon', type=float, default=0.8)
@click.option('--delta', type=float, default=0.2)
@click.pass_obj
def count_tweakeys(obj: GlobalArgs, epsilon: float, delta: float) -> None|int:
    """count valid tweakeys using ApproxMC"""
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=True, needs_tweak=True)
    ensure_executables('approxmc')
    cipher.count_tweakey_space(epsilon, delta, count_key=True, count_tweak=True)

@cli.command()
@click.option('--epsilon', type=float, default=0.8)
@click.option('--delta', type=float, default=0.2)
@click.pass_obj
def count_keys(obj: GlobalArgs, epsilon: float, delta: float) -> None:
    """count valid keys using ApproxMC"""
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=True, needs_tweak=False)
    ensure_executables('approxmc')
    cipher.count_tweakey_space(epsilon, delta, count_key=True, count_tweak=False)

@cli.command()
@click.option('--epsilon', type=float, default=0.8)
@click.option('--delta', type=float, default=0.2)
@click.pass_obj
def count_tweaks(obj: GlobalArgs, epsilon: float, delta: float) -> None:
    """count valid tweaks using ApproxMC"""
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=False, needs_tweak=True)
    cipher.count_tweakey_space(epsilon, delta, count_key=False, count_tweak=True)


@cli.command()
@click.pass_obj
def count_tweakeys_lin(obj: GlobalArgs) -> None:
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=True, needs_tweak=True)
    ensure_executables('cryptominisat5')
    cipher.count_lin_tweakey_space(count_key=True, count_tweak=True)

@cli.command()
@click.pass_obj
def count_keys_lin(obj: GlobalArgs) -> None:
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=True, needs_tweak=False)
    ensure_executables('cryptominisat5')
    cipher.count_lin_tweakey_space(count_key=True, count_tweak=False)

@cli.command()
@click.pass_obj
def count_tweaks_lin(obj: GlobalArgs) -> None:
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=False, needs_tweak=True)
    ensure_executables('cryptominisat5')
    cipher.count_lin_tweakey_space(count_key=False, count_tweak=True)


@cli.command()
@click.option('--trials', type=int, default=1_000)
@click.pass_obj
def count_tweakeys_sat(obj: GlobalArgs, trials: int) -> None:
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=True, needs_tweak=True)
    cipher.count_tweakey_space_sat_solver(trials, count_key=True, count_tweak=True)

@cli.command()
@click.option('--trials', type=int, default=1_000)
@click.pass_obj
def count_keys_sat(obj: GlobalArgs, trials: int) -> None:
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=True, needs_tweak=False)
    cipher.count_tweakey_space_sat_solver(trials, count_key=True, count_tweak=False)

@cli.command()
@click.option('--trials', type=int, default=1_000)
@click.pass_obj
def count_tweaks_sat(obj: GlobalArgs, trials: int) -> None:
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=False, needs_tweak=True)
    cipher.count_tweakey_space_sat_solver(trials, count_key=False, count_tweak=True)


@cli.command()
@click.option('--epsilon', type=float, default=0.8)
@click.option('--delta', type=float, default=0.2)
@click.option('--fixed-key', is_flag=True)
@click.option('--fixed-tweak', is_flag=True)
@click.pass_obj
def count_prob(obj: GlobalArgs, epsilon: float, delta: float, fixed_key: bool, fixed_tweak: bool) -> None:
    cipher = obj.cipher
    ensure_cipher_comatible(cipher, needs_key=fixed_key, needs_tweak=fixed_tweak)
    cipher.count_probability(epsilon, delta, fixed_key=fixed_key, fixed_tweak=fixed_tweak)


@cli.command()
@click.pass_obj
def solve(obj: GlobalArgs) -> None:
    cipher = obj.cipher
    try:
        cipher.solve()
    except UnsatException:
        pass

@cli.command()
@click.pass_obj
def find_conflict(obj: GlobalArgs) -> None:
    """list s-boxes which lead to an impossible characteristic"""
    cipher = obj.cipher
    if not cipher.model_sbox_assumptions:
        raise click.UsageError("command 'find-conflict' requires --sbox-assumptions")
    cipher.find_conflict()

@cli.command()
@click.argument('filename', type=click.Path(writable=True))
@click.pass_obj
def write_cnf(obj: GlobalArgs, filename: Path) -> None:
    with open(filename, 'w') as f:
        log.info(f"writing CNF to {filename}")
        f.write(obj.cipher.cnf.to_dimacs())


@cli.command()
@click.pass_obj
def embed(obj: GlobalArgs) -> None:
    cipher = obj.cipher
    characteristic = obj.characteristic
    sys.argv = sys.argv[:1] # remove all arguments except the command, so start_ipython doesn't try to parse it
    start_ipython(user_ns=globals()|locals())


if __name__ == "__main__":
    cli()
