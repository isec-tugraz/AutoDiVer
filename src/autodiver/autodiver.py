#!/usr/bin/env python3
from __future__ import annotations

import json
import logging.config
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess as sp
import sys
from typing import Optional, Literal, TYPE_CHECKING
import numpy as np
from .util import create_latex, unique_path
import click

from autodiver import version
from autodiver.autodiver_types import ModelType, UnsatException, RoundMode, SearchMode, CARD_ENC_MAP

if TYPE_CHECKING:
    from autodiver.cipher_model import SboxCipher

from autodiver.cipher_model import DifferentialCharacteristic


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

_ciphers: dict[str, tuple[str, str, str]] = {
    "ascon": ("autodiver.ascon.ascon_model", "Ascon", "AsconCharacteristic"),
    "gift64": ("autodiver.gift.gift_model", "Gift64", "Gift64Characteristic"),
    "gift64-full-key": ("autodiver.gift.gift_model", "Gift64FullKey", "Gift64Characteristic"),
    "gift128": ("autodiver.gift.gift_model", "Gift128", "Gift128Characteristic"),
    "gift128-full-key": ("autodiver.gift.gift_model", "Gift128FullKey", "Gift128Characteristic"),
    "midori64": ("autodiver.midori64.midori64_model", "Midori64", "Midori64Characteristic"),
    "midori64-long-key": ("autodiver.midori64.midori64_model", "Midori64LongKey", "Midori64Characteristic"),
    "midori128": ("autodiver.midori128.midori128_model", "Midori128", "Midori128Characteristic"),
    "midori128-long-key": ("autodiver.midori128.midori128_model", "Midori128LongKey", "Midori128Characteristic"),
    "present80": ("autodiver.present.present_model", "Present80", "PresentCharacteristic"),
    "present-long-key": ("autodiver.present.present_model", "PresentLongKey", "PresentCharacteristic"),
    "pyjamask": ("autodiver.pyjamask.pyjamask96_model", "Pyjamask_with_Keyschedule", "Pyjamask96Characteristic"),
    "pyjamask-long-key": ("autodiver.pyjamask.pyjamask96_model", "Pyjamask_Longkey", "Pyjamask96Characteristic"),
    "rectangle128": ("autodiver.rectangle128.rectangle_model", "Rectangle128", "RectangleCharacteristic"),
    "rectangle-long-key": ("autodiver.rectangle128.rectangle_model", "RectangleLongKey", "RectangleCharacteristic"),
    "skinny64": ("autodiver.skinny.skinny_model", "Skinny64", "Skinny64Characteristic"),
    "skinny64-long-key": ("autodiver.skinny.skinny_model", "Skinny64LongKey", "Skinny64Characteristic"),
    "skinny128": ("autodiver.skinny.skinny_model", "Skinny128", "Skinny128Characteristic"),
    "skinny128-long-key": ("autodiver.skinny.skinny_model", "Skinny128LongKey", "Skinny128Characteristic"),
    "speck32-long-key": ("autodiver.speck.speck_model", "Speck32LongKey", "Speck32Characteristic"),
    "speck48-long-key": ("autodiver.speck.speck_model", "Speck48LongKey", "Speck48Characteristic"),
    "speck64-long-key": ("autodiver.speck.speck_model", "Speck64LongKey", "Speck64Characteristic"),
    "speck96-long-key": ("autodiver.speck.speck_model", "Speck96LongKey", "Speck96Characteristic"),
    "speck128-long-key": ("autodiver.speck.speck_model", "Speck128LongKey", "Speck128Characteristic"),
    "speedy192": ("autodiver.speedy192.speedy192_model", "Speedy192", "Speedy192Characteristic"),
    "warp": ("autodiver.warp128.warp128_model", "WARP128", "WarpCharacteristic"),
}


@click.group()
@click.argument('cipher_name', type=click.Choice(list(_ciphers.keys())), required=True)
@click.argument('characteristic_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True)
@click.option('--sbox-assumptions', is_flag=True, help="add assumption variables for all S-boxes")
@click.option('--model-type', type=click.Choice([mt.value for mt in ModelType]), default=ModelType.solution_set.value, help="using split-solution set allows for more efficient but less accurate modeling")
@click.option('--rounds-from-to', nargs=2, type=int, help='For example: 2 4 - use rounds 2 to 4 (3 rounds) of this characteristic.')
@click.pass_context
def cli(ctx, cipher_name: str, characteristic_path: str|Path, sbox_assumptions: bool, model_type: str, rounds_from_to: tuple[int, int]) -> None:
    characteristic_path = Path(characteristic_path)
    setup_logging(characteristic_path.with_suffix('.jsonl'))
    git_cmd = shutil.which('git')
    git_commit = git_cmd and sp.check_output([git_cmd, 'rev-parse', 'HEAD']).decode().strip()
    git_changed_files = git_cmd and sp.check_output([git_cmd, 'status', '--porcelain', '-uno', '-z']).decode().strip('\0').split('\0')
    log.info(f"version: {version}, git_commit: {git_commit}, git_changed_files: {git_changed_files}")
    log.debug("arguments: %s", sys.argv, extra={"cli_args": sys.argv, "git_commit": git_commit, "git_changed_files": git_changed_files, "version": version})

    module_name, cipher_type_name, characteristic_type_name = _ciphers[cipher_name]
    import importlib
    module = importlib.import_module(module_name)
    Cipher: type[SboxCipher] = getattr(module, cipher_type_name)
    Characteristic: type[DifferentialCharacteristic] = getattr(module, characteristic_type_name)

    characteristic = Characteristic.load(characteristic_path)
    if rounds_from_to is not None:
        characteristic.truncate_rounds(rounds_from_to)
    cipher = Cipher(characteristic, model_sbox_assumptions=sbox_assumptions, model_type=ModelType(model_type))
    ddt_prob_log2 = characteristic.log2_ddt_probability()
    relative_characteristic_path = characteristic_path.relative_to(Path(".").resolve())
    log.info(f"loaded characteristic with {characteristic.num_rounds} rounds from {relative_characteristic_path} with ddt probability 2**{ddt_prob_log2:.1f}")

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

def ensure_cipher_compatible(cipher: SboxCipher, kind: Literal['key', 'tweak', 'tweakey']) -> None:
    if cipher.key_size == 0 and 'key' in kind:
        raise click.UsageError(f"{cipher.cipher_name} has no key")
    if cipher.tweak_size == 0 and 'tweak' in kind:
        raise click.UsageError(f"{cipher.cipher_name} has no tweak")

def default_kind(cipher: SboxCipher) -> Literal['key', 'tweak', 'tweakey']:
    if cipher.key_size > 0 and cipher.tweak_size > 0:
        return 'tweakey'
    if cipher.key_size > 0:
        return 'key'
    if cipher.tweak_size > 0:
        return 'tweak'
    raise click.UsageError(f"{cipher.cipher_name} has neither key nor tweak")

@cli.command()
@click.option('--epsilon', type=float, default=0.8)
@click.option('--delta', type=float, default=0.2)
@click.option('-k', '--kind', type=click.Choice(['tweakey', 'key', 'tweak']), default=None)
@click.pass_obj
def count_tweakeys(obj: GlobalArgs, epsilon: float, delta: float, kind: Literal['key', 'tweak', 'tweakey']|None) -> None:
    """count valid tweakeys using ApproxMC"""
    cipher = obj.cipher

    if kind is None:
        kind = default_kind(cipher)

    ensure_cipher_compatible(cipher, kind)
    cipher.count_tweakey_space(epsilon, delta, kind=kind)


@cli.command()
@click.pass_obj
@click.option('-k', '--kind', type=click.Choice(['tweakey', 'key', 'tweak']), default=None)
@click.option('--explain', is_flag=True)
def count_tweakeys_lin(obj: GlobalArgs, kind: Literal['key', 'tweak', 'tweakey']|None, explain: bool) -> None:
    """find the affine hull of the set of valid tweakeys"""
    cipher = obj.cipher

    if explain and not cipher.model_sbox_assumptions:
        raise click.UsageError("option '--explain' requires --sbox-assumptions")

    if kind is None:
        kind = default_kind(cipher)

    ensure_cipher_compatible(cipher, kind)
    ensure_executables('cryptominisat5')
    cipher.find_affine_hull(kind)

    if explain:
        cipher.explain_affine_hull(kind)

@cli.command()
@click.option('-k', '--kind', type=click.Choice(['tweakey', 'key', 'tweak']), default=None)
@click.option('-n', '--trials', type=int, default=1_000, help="number of tweakeys to test")
@click.option('-m', '--max-clause-len', type=int, default=20, help="maximum length of clauses")
@click.pass_obj
def count_tweakeys_sat(obj: GlobalArgs, trials: int, kind: Literal['key', 'tweak', 'tweakey']|None, max_clause_len: int) -> None:
    """estimate size of valid tweakey space experimentally with SAT solvers"""
    cipher = obj.cipher

    if kind is None:
        kind = default_kind(cipher)

    ensure_cipher_compatible(cipher, kind)
    cipher.count_tweakey_space_sat_solver(trials, kind, max_clause_len=max_clause_len)


@cli.command()
@click.pass_obj
@click.option('-k', '--kind', type=click.Choice(['tweakey', 'key', 'tweak']), default=None)
@click.option('-n', '--trials', type=int, default=1_000, help="number of tweakeys to test")
@click.option('-m', '--max-clause-len', type=int, default=20, help="maximum length of clauses")
@click.option('--explain', is_flag=True)
def count_tweakeys_combined(obj: GlobalArgs, kind: Literal['key', 'tweak', 'tweakey']|None, trials: int, max_clause_len: int, explain: bool) -> None:
    """find the affine hull and verify the remaining keyspace experimentally with SAT solvers"""
    cipher = obj.cipher

    if kind is None:
        kind = default_kind(cipher)

    ensure_cipher_compatible(cipher, kind)
    ensure_executables('cryptominisat5')

    cipher.find_affine_hull(kind)
    if explain:
        cipher.explain_affine_hull(kind)
    cipher.count_tweakey_space_sat_solver(trials, kind, use_affine_hull=True, max_clause_len=max_clause_len, explain=explain)


@cli.command()
@click.option('--epsilon', type=float, default=0.8)
@click.option('--delta', type=float, default=0.2)
@click.option('--fixed-key', is_flag=True)
@click.option('--fixed-tweak', is_flag=True)
@click.pass_obj
def count_prob(obj: GlobalArgs, epsilon: float, delta: float, fixed_key: bool, fixed_tweak: bool) -> None:
    """estimate the probability using ApproxMC"""
    cipher = obj.cipher
    if fixed_key and cipher.key_size == 0:
        raise click.UsageError(f"{cipher.cipher_name} has no key")
    if fixed_tweak and cipher.tweak_size == 0:
        raise click.UsageError(f"{cipher.cipher_name} has no tweak")
    cipher.count_probability(epsilon, delta, fixed_key=fixed_key, fixed_tweak=fixed_tweak)


@cli.command()
@click.pass_obj
def solve(obj: GlobalArgs) -> None:
    """find a satisfying pair for the characteristic"""
    cipher = obj.cipher
    try:
        cipher.solve()
    except UnsatException:
        pass

@cli.command()
@click.pass_obj
def find_conflicts(obj: GlobalArgs) -> None:
    """list S-boxes which lead to a contradiction"""
    cipher = obj.cipher
    if not cipher.model_sbox_assumptions:
        raise click.UsageError("command 'find-conflict' requires --sbox-assumptions")
    cipher.find_conflicts()

@cli.command()
@click.argument('filename', type=click.Path(writable=True))
@click.option('--convert-xors', is_flag=True, help="convert XOR constraints to CNF")
@click.option('--assumptions/--no-assumptions', default=True, is_flag=True, help="add unit clausses for S-box assumptions (default: True)")
@click.pass_obj
def write_cnf(obj: GlobalArgs, filename: Path, convert_xors: bool, assumptions: bool) -> None:
    """write the CNF to a file"""
    from sat_toolkit.formula import CNF

    if convert_xors:
        cnf = obj.cipher.cnf.to_cnf()
    else:
        cnf = obj.cipher.cnf

    cnf = cnf.copy()
    if assumptions and obj.cipher.model_sbox_assumptions:
        for assumption_var in obj.cipher.sbox_assumptions.flatten():
            cnf += CNF([assumption_var, 0])

    with open(filename, 'w') as f:
        log.info(f"writing CNF to {filename}")
        f.write(cnf.to_dimacs())


@cli.command()
@click.pass_obj
def embed(obj: GlobalArgs) -> None:
    """launch an interactive IPython shell"""
    cipher = obj.cipher
    characteristic = obj.characteristic

    try:
        from IPython import start_ipython
    except ImportError:
        click.echo("Error: optional dependency IPython is required for this command", err=True)
        sys.exit(1)

    start_ipython(argv=[], user_ns=globals()|locals())


_ciphers_char_search: dict[str, tuple[str, str, str, bool]] = {
    "ascon": ("autodiver.ascon.ascon_model", "Ascon", "AsconCharacteristic", False),
    "gift64": ("autodiver.gift.gift_model", "Gift64", "Gift64Characteristic", True),
    "gift128": ("autodiver.gift.gift_model", "Gift128", "Gift128Characteristic", True),
    "midori64": ("autodiver.midori64.midori64_model", "Midori64", "Midori64Characteristic", False),
    "midori128": ("autodiver.midori128.midori128_model", "Midori128", "Midori128Characteristic", False),
    "present80": ("autodiver.present.present_model", "Present80", "PresentCharacteristic", True),
    "pyjamask96": ("autodiver.pyjamask.pyjamask96_model", "Pyjamask_with_Keyschedule", "Pyjamask96Characteristic", False),
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

_tikzify_supported = sorted(
    name for name, (_, _, _, supports_tikz) in _ciphers_char_search.items()
    if supports_tikz
)

_tikzify_help = (
    "Visualize the found characteristic in LaTeX. "
    f"Supported for: {', '.join(_tikzify_supported)}"
)

# choose lower third of table so speed is still feasible, but also the area we care more about - (where it takes the longest)
ciphers_round_number: dict[str, (int, int)] = {
    "gift64" : (10, 47),
    "gift128" : (9, 45),
    "midori64" : (5, 45),
    "midori128" : (5, 48),
    "present80" : (11, 45),
    "rectangle128" : (9, 35),
    "skinny64" : (5, 23),
    "skinny128" : (11, 103),
    "speck32" : (8, 23),
    "speck64" : (7, 20),
    "speck128" : (7, 20),
    "warp" : (13, 67)
}

def profile_cardenc() -> None:
    for cipher in ciphers_round_number:
        print(cipher)
        for card_enc in CARD_ENC_MAP:
            if card_enc in ["pairwise", "bitwise", "ladder", "native"]: # don't support our usecase
                continue
            run_search_characteristic(cipher, ciphers_round_number[cipher][0], tikzify=False, seed=7, log_probability=ciphers_round_number[cipher][1], rounding_mode=RoundMode.DOWN.value, searching_mode=SearchMode.UPWARDS.value, save=False, related_tweak=False, card_enc=card_enc, save_perf=True)

    # loop over all ciphers and all cardinality encodings here
    # search for all ciphers for 5 rounds
    # save time_sat_search, time_unsat_search, encoding, cipher in npz file in /misc/card_enc/<cipher>/<enc>_stat.npz ? and then make separate script to print results to latex
    # also save # vars and # clauses just in case



@click.command()
@click.argument('cipher_name', type=click.Choice(list(_ciphers_char_search.keys())), required=True)
@click.argument('num-rounds', nargs=1, type=int, required=True)
@click.option("--tikzify", is_flag=True, help=_tikzify_help)
@click.option("--seed", type=int, default=None)
@click.option("--log-probability", type=int, default=None, help="the minimum probability we are searching for, expressed as log2(p)")
@click.option("--rounding-mode",type=click.Choice([m.value for m in RoundMode]), default=RoundMode.DOWN.value)
@click.option("--searching-mode",type=click.Choice([m.value for m in SearchMode]), default=SearchMode.BINARY.value)
@click.option("--no-save", is_flag=True, help="don't save this characteristic")
@click.option("--related-tweak", is_flag=True, help="will execute the related tweak search; currently only available for SKINNY")
@click.option("--card-enc", type=click.Choice(sorted(CARD_ENC_MAP)), default="kmtotalizer")
def search_characteristic(cipher_name: str, num_rounds: int, tikzify: bool, seed: int, log_probability: int, rounding_mode: RoundMode, searching_mode: SearchMode, no_save: bool, related_tweak: bool, card_enc: str) -> None:
    run_search_characteristic(cipher_name, num_rounds, tikzify, seed, log_probability, rounding_mode, searching_mode, not no_save, related_tweak, card_enc, False)


def run_search_characteristic(cipher_name: str, num_rounds: int, tikzify: bool, seed: int, log_probability: int, rounding_mode: RoundMode, searching_mode: SearchMode, save: bool, related_tweak: bool, card_enc: str, save_perf: bool) -> int:
    """search for a characteristic for the given cipher"""
    directory = Path.cwd() / "logfiles"
    directory.mkdir(parents=True, exist_ok=True)
    setup_logging('logfiles/search_char' + cipher_name + '_' + str(num_rounds) + '.jsonl')

    git_cmd = shutil.which('git')
    git_commit = git_cmd and sp.check_output([git_cmd, 'rev-parse', 'HEAD']).decode().strip()
    git_changed_files = git_cmd and sp.check_output([git_cmd, 'status', '--porcelain', '-uno', '-z']).decode().strip(
        '\0').split('\0')
    log.info(f"version: {version}, git_commit: {git_commit}, git_changed_files: {git_changed_files}")
    log.debug("arguments: %s", sys.argv,
              extra={"cli_args": sys.argv, "git_commit": git_commit, "git_changed_files": git_changed_files,
                     "version": version})

    if related_tweak and not cipher_name in ["skinny64", "skinny128"]:
        print(f"related-tweak search is only available for skinny, not for {cipher_name}")
        exit(0)

    module_name, cipher_type_name, characteristic_type_name, _ = _ciphers_char_search[cipher_name]

    import importlib
    module = importlib.import_module(module_name)
    Cipher: type[SboxCipher] = getattr(module, cipher_type_name)
    Characteristic: type[DifferentialCharacteristic] = getattr(module, characteristic_type_name)


    characteristic = Characteristic.load_empty_characteristic(num_rounds) # pro forma characteristic; could be used to indicate active sboxes if we wanted


    cipher = Cipher(characteristic, search_char=True, related_tweak=related_tweak, rounding_mode=RoundMode(rounding_mode), searching_mode=SearchMode(searching_mode), log_prob=log_probability, card_enc=CARD_ENC_MAP[card_enc])

    try:
        model = cipher.solve(seed=seed)

        characteristic = Characteristic.load_from_model(model) # actual recovered characteristic
        log_probability = characteristic.log2_ddt_probability()
        print(f"probability: {log_probability}")

        if save:
            directory = Path.cwd() / "found_trails" / str(cipher_name + ("_rel_tweak" if related_tweak else ""))
            directory.mkdir(parents=True, exist_ok=True)
            char_path = Path(Path.cwd() / "found_trails" / (cipher_name + ("_rel_tweak_xeon" if related_tweak else "")) /(cipher_name  + "_r" + str(num_rounds))).with_suffix('.npz')
            characteristic.save_npz(unique_path(char_path), cipher_name, num_rounds, log_probability, cipher.stat_sat_search, cipher.log_prob, cipher.rounding_mode.value)

        if save_perf:
            dir_name = "misc/card_enc_var_round_fixed_seed"
            directory = Path.cwd() / dir_name / cipher_name
            directory.mkdir(parents=True, exist_ok=True)
            char_path = Path(Path.cwd() / dir_name / cipher_name / card_enc).with_suffix('.npz')
            np.savez(char_path, stat_sat_search=cipher.stat_sat_search, stat_unsat_search=cipher.stat_unsat_search, num_rounds=num_rounds, boundary=cipher.log_prob)

        if tikzify:
            create_latex(characteristic)

        return -log_probability


    except UnsatException:
        pass


@click.command()
@click.argument('cipher_name', type=click.Choice(list(_ciphers_char_search.keys())), required=True)
@click.option("--rounding-mode",type=click.Choice([m.value for m in RoundMode]), default=RoundMode.DOWN.value)
@click.option("--related-tweak", is_flag=True, help="will execute the related key search; currently only available for SKINNY")
def search_characteristic_all(cipher_name: str, rounding_mode: RoundMode, related_tweak: bool) -> None:
    """search for a characteristic for the given cipher"""
    num_rounds = 1
    probability = 0
    while True:
        probability = run_search_characteristic(cipher_name, num_rounds, tikzify=False, seed=None, log_probability=probability, rounding_mode=RoundMode(rounding_mode), searching_mode=SearchMode.BINARY, save=True, related_tweak=related_tweak, save_perf=False)
        num_rounds += 1


if __name__ == "__main__":
    cli()
