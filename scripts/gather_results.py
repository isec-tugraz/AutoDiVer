#!/usr/bin/env python3
"""
Script to gather results from multiple log files and output them as a markdown file
"""
import argparse
from collections import defaultdict
import json
import sys
from math import log2, sqrt
from datetime import datetime
from typing import TextIO
import re

import scipy.stats as stats
import tabulate

from pathlib import Path

def wilson_score_interval(x, n, confidence):
    """
    Calculate the Wilson score interval for a given observed occurrences, total observations, and confidence level.

    Parameters:
    x (int): The number of observed occurrences.
    n (int): The total number of occurrences.
    confidence (float): The desired confidence level (between 0 and 1).

    Returns:
    tuple: A tuple containing the lower and upper bounds of the Wilson score interval.
    """

    # Calculate the observed proportion
    observed_proportion = x / n

    # Calculate the z-score corresponding to the confidence level
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    # Calculate the term under the square root
    term = z * sqrt((observed_proportion * (1 - observed_proportion) / n) + (z**2 / (4 * n**2)))

    # Calculate the lower and upper bounds of the interval
    lower_bound = (observed_proportion + (z**2) / (2 * n) - term) / (1 + (z**2) / n)
    upper_bound = (observed_proportion + (z**2) / (2 * n) + term) / (1 + (z**2) / n)

    return lower_bound, upper_bound


def find_log_files(path: Path):
    """
    find and yield all .jsonl files in the given directory and its subdirectories
    """
    for file in sorted(path.iterdir()):
        if file.name.startswith('.'):
            continue
        if file.name in ['__pycache__', 'venv', 'build']:
            continue
        elif file.is_dir():
            yield from find_log_files(file)
        elif file.is_file() and file.suffix == '.jsonl':
            yield file


def find_results_in_file(file: Path):
    with file.open('r') as f:
        for line_number, line in enumerate(f, start=1):
            if 'context' not in line or 'RESULT' not in line:
                continue
            data = json.loads(line)
            if 'context' not in data:
                continue
            yield line_number, data


def format_time(t: float):
    """format time as hours, minutes, seconds """
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    milliseconds = int((t % 1) * 1000)

    parts = []
    if hours:
        parts.append(f'{hours}h')
    if minutes:
        parts.append(f'{minutes}m')
    if seconds:
        parts.append(f'{seconds}s')

    if not hours and not minutes:
        parts.append(f'{milliseconds}ms')

    return ' '.join(parts)

def fmt_log2(p: float, digits=2, latex=False):
    if p == 0:
        return str(p)
    if p < 0:
        return '-' + fmt_log2(-p, latex=latex)

    assert p > 0
    if latex:
        return '$2^{' + f'{log2(p):.{digits}f}' + '}$'
    return f'2^{log2(p):.{digits}f}'

def fmt_ci_latex(lower: float, upper: float):
    if lower > 0 and upper > 0:
        return '$2^{[' + f'{log2(lower):.2f},{log2(upper):.2f}' + ']}$'
    return "???"

def fmt_ci_latex_log2(lower: float, upper: float):
    return '$2^{[' + f'{lower:.2f},{upper:.2f}' + ']}$'



def gather_results(argv: list[str], md_file: TextIO, tex_file: TextIO, prob_file: TextIO, prob_tex_file: TextIO):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, nargs='?', default=Path('.'), help='Path to the directory containing the log files')
    # parser.add_argument('output', type=Path, nargs='?', default=Path('results.md'), help='Path to the output markdown file')
    args = parser.parse_args(argv)

    solve_results = []
    count_results = []
    count_tweakey_results = []
    count_tweakey_lin_results = []
    count_tweakey_sat_results = {}

    KEY_COUNT_LIN = 'LC'
    KEY_COUNT_SAT = 'exp.\\ est.\\'
    KEY_COUNT = 'AMC'
    ROUNDS = r'\#R'
    DDT_PROB = r'EDP'

    latex_table = defaultdict(lambda: {ROUNDS: '??', DDT_PROB: r'$2^{-??}$', KEY_COUNT: None, KEY_COUNT_SAT: None, KEY_COUNT_LIN: None})


    lin_constraints = []
    base_path = Path('.').resolve()

    for log_file in find_log_files(args.path):
        for line_number, result in find_results_in_file(log_file):
            trail = Path(result['context']['char']['file_path'])
            rounds_from_to: tuple[int, int]|None = result['context']['char'].get('rounds_from_to')
            if rounds_from_to is not None:
                rounds_from_to = (rounds_from_to[0], rounds_from_to[1])
            log2_ddt_probability: float|None = result['context']['char'].get('log2_ddt_prob')
            cwd = result['context'].get('cwd')

            if cwd:
                cwd = Path(cwd)

            if 'autodiver' in trail.parts:
                trail = Path(*trail.parts[trail.parts.index('autodiver')+1:])
            elif trail.is_absolute():
                if trail.parts[:3] == ('/', 'mnt', 'trails'):
                    trail = Path(*trail.parts[2:])
                else:
                    trail = trail.relative_to(cwd or base_path)

            if trail.parts[0] == 'autodiver':
                trail = Path(*trail.parts[1:])

            cipher = result['context']['cipher']
            model_type = result['context'].get('model_type', 'solution_set')
            timestamp = datetime.fromisoformat(result['timestamp'])

            original_trail = trail
            numrounds = result['context']['char'].get('numrounds')
            if rounds_from_to is not None:
                from_rnd, to_rnd = rounds_from_to
                trail = trail.with_stem(f'{trail.stem}_rnd_{from_rnd}-{to_rnd}')
                numrounds = to_rnd - from_rnd + 1


            if model_type not in ('ModelType.solution_set', 'solution_set'):
                continue

            try:
                if 'key_size' in result['context'] and 'tweak_size' in result['context']:
                    key_size = result['context']['key_size']
                    tweak_size = result['context']['tweak_size']
                else:
                    (key_size, tweak_size) = {
                        'gift64': (128, 0),
                        'gift128': (128, 0),
                        'midori64': (128, 0),
                        'midori128': (128, 0),
                        'warp128': (128, 0),
                        'ascon': (0, 0),
                        'speedy192': (192, 0),
                        'skinny64': (64, 128),
                        'skinny128': (128, 256),
                        'present80': (80, 0),
                        'rectangle128': (128, 0),
                        'presentlongkey': (0, 0),
                        # 'skinnylongkey': (0, 0),
                    }[cipher.lower()]
                tweakey_size = key_size + tweak_size
                has_tweak = tweak_size > 0

                if 'solve_result' in result:
                    key = result['solve_result']['key']
                    tweak = result['solve_result']['tweak']
                    pt = result['solve_result']['pt']
                    time = format_time(result['solve_result']['time'])

                    solve_result = {'cipher': cipher, 'trail': trail, 'key': key, 'tweak': tweak, 'pt': pt, 'time': time}
                    solve_results.append({k: v for k, v in solve_result.items() if v != ''})

                if 'count_result' in result:
                    probability = result['count_result']['probability']
                    key = result['count_result']['key']
                    tweak = result['count_result']['tweak']
                    time = format_time(result['count_result']['time'])
                    if rounds_from_to is not None:
                        rounds = f"{rounds_from_to[0]} - {rounds_from_to[1]}"
                    else:
                        rounds = '-'

                    epsilon = result['count_result']['epsilon']
                    delta = result['count_result']['delta']

                    prob_str = f'{fmt_log2(probability)}'

                    count_result = {'cipher': cipher, 'trail': original_trail, 'rounds': rounds_from_to, 'measured_prob': prob_str, 'stated_prob': f"2^{log2_ddt_probability}", 'delta': delta, 'epsilon': epsilon, 'key': key, 'tweak': tweak, 'time': time}
                    count_results.append({k: v for k, v in count_result.items() if v != ''})

                if 'count_tweakey_result' in result:
                    num_keys = result['count_tweakey_result']['num_keys']
                    count_key = result['count_tweakey_result']['count_key']
                    count_tweak = result['count_tweakey_result']['count_tweak'] and has_tweak
                    time = format_time(result['count_tweakey_result']['time'])

                    kind = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]

                    epsilon = result['count_tweakey_result']['epsilon']
                    delta = result['count_tweakey_result']['delta']

                    num_keys_str = f'{fmt_log2(num_keys)}'

                    count_tweakey_result = {'cipher': cipher, 'trail': trail, 'kind': kind, 'count': num_keys_str, 'delta': delta, 'epsilon': epsilon, 'time': time}
                    count_tweakey_results.append({k: v for k, v in count_tweakey_result.items() if v != ''})

                    latex_table[(trail, kind)][KEY_COUNT] = fmt_log2(num_keys, latex=True)


                if 'count_tweakey_lin_result' in result:
                    constraints = result['count_tweakey_lin_result']['constraints']
                    count_key = result['count_tweakey_lin_result']['count_key']
                    count_tweak = result['count_tweakey_lin_result']['count_tweak'] and has_tweak
                    time = format_time(result['count_tweakey_lin_result']['time'])

                    kind = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]

                    count_tweakey_lin_result = {'cipher': cipher, 'trail': trail, 'kind': kind, 'constraints': len(constraints), 'time': time}
                    count_tweakey_lin_results.append({k: v for k, v in count_tweakey_lin_result.items() if v != ''})

                    latex_table[(trail, kind)][KEY_COUNT_LIN] = f'{len(constraints)}'

                    if len(constraints) > 0:
                        iter_constraints = iter(constraints)

                        first = next(iter_constraints)
                        lin_constraints.append({'cipher': cipher, 'trail': trail, 'kind': kind, 'constraint': first})

                        for constraint in iter_constraints:
                            lin_constraints.append({'cipher': cipher, 'trail': '', 'kind': '', 'constraint': constraint})
                        lin_constraints.append({})

                if 'count_tweakeys_sat_result' in result:
                    if timestamp < datetime.fromisoformat('2024-07-15 17:25:13 +0200'):
                        continue # skip buggy results

                    # these results are restricted to the affine hull of valid keys
                    if 'use_affine_hull' in result and result['use_affine_hull']:
                        continue

                    trials = result['count_tweakeys_sat_result']['trials']
                    count_sat = result['count_tweakeys_sat_result']['count_sat']
                    count_key = result['count_tweakeys_sat_result']['count_key']
                    count_tweak = result['count_tweakeys_sat_result']['count_tweak'] and has_tweak
                    time = result['count_tweakeys_sat_result']['time']
                    kind = [None, 'tweak', 'key', 'tweakey'][2*count_key + count_tweak]

                    key = (cipher, trail, kind)
                    if key in count_tweakey_sat_results:
                        count_tweakey_sat_results[key]['trials'] += trials
                        count_tweakey_sat_results[key]['count_sat'] += count_sat
                        count_tweakey_sat_results[key]['time'] += time
                    else:
                        count_tweakey_sat_results[key] = {
                            'cipher': cipher,
                            'trail': trail,
                            'kind': kind,
                            'trials': trials,
                            'count_sat': count_sat,
                            'time': time,
                            'tweakey conditions': set(),
                            '_tweakey_size': tweakey_size,
                        }

                if 'key_conditions_sat_result' in result:
                    tweakey_conditions = set(result['key_conditions_sat_result']['key_conditions'])
                    kind = result['key_conditions_sat_result']['kind']

                    key = (cipher, trail, kind)
                    count_tweakey_sat_results[key]['tweakey conditions'].update(tweakey_conditions)

                for kind in ('tweak', 'key', 'tweakey'):
                    key = (trail, kind)
                    if key in latex_table and numrounds is not None:
                        latex_table[key][ROUNDS] = f"{numrounds}"

            except Exception as e:
                print(f'Could not process {log_file}:{line_number}: {e!r}', file=sys.stderr)



    # if solve_results:
    #     print(f'# Solve Results', file=md_file)
    #     print(file=md_file)
    #     print(tabulate.tabulate(solve_results, headers='keys', tablefmt='github') + '\n', file=md_file)
    #     print(file=md_file)

    if count_results:
        print(f'# Count Probability Results', file=md_file)
        print(file=md_file)
        print(tabulate.tabulate(count_results, headers='keys', tablefmt='github') + '\n', file=md_file)
        print(file=md_file)

        prob_results_by_trail = defaultdict(list)
        for r in count_results:
            cipher, trail = r['cipher'], r['trail']
            delta, epsilon = r['delta'], r['epsilon']
            r_copy = r.copy()
            del r_copy['cipher']
            del r_copy['trail']
            del r_copy['delta']
            del r_copy['epsilon']
            prob_results_by_trail[cipher, trail, delta, epsilon].append(r_copy)

        for (cipher, trail, delta, epsilon), results in prob_results_by_trail.items():
            print(f'## {cipher} - {trail} (delta={delta}, epsilon={epsilon}), ', file=prob_file)
            print(f'%% {cipher} - {trail} (delta={delta}, epsilon={epsilon}), ', file=prob_tex_file)


            possible_rounds = {r['rounds'] for r in results}
            possible_rounds = sorted({r[1] - r[0] + 1 for r in possible_rounds if r is not None}, reverse=True)

            for relevant_rounds in possible_rounds:

                relevant_results = [r for r in results if r['rounds'] is not None and r['rounds'][1] - r['rounds'][0] + 1 == relevant_rounds]
                relevant_results = sorted(relevant_results, key=lambda r: r['rounds'])
                relevant_results_dict = {}
                for r in relevant_results:
                    relevant_results_dict[r['rounds']] = r

                rounds = ['Rounds']
                stated_probs = ['Stated prob.']
                measured_probs = ['Measured prob.']
                times = ['Time']
                for r in relevant_results_dict.values():
                    rounds.append(r['rounds'])
                    stated_probs.append(r['stated_prob'])
                    measured_probs.append(r['measured_prob'])
                    times.append(r['time'])

                print(file=prob_file)
                print(tabulate.tabulate([rounds, stated_probs, measured_probs, times], headers='firstrow', tablefmt='github') + '\n', file=prob_file)

                print(f'% Rounds: {relevant_rounds}', file=prob_tex_file)
                print("\\begin{table}", file=prob_tex_file)
                print("\\scriptsize", file=prob_tex_file)
                print("\\centering", file=prob_tex_file)

                for i in range(1, len(stated_probs)):
                    stated_probs[i] = re.sub(r'2\^-(\d+\.\d*)', r'$\1$', stated_probs[i])
                    measured_probs[i] = re.sub(r'2\^-(\d+\.\d*)', r'$\1$', measured_probs[i])
                    rounds[i] = f"{rounds[i][0]}--{rounds[i][1]}"

                print(tabulate.tabulate([rounds, stated_probs, measured_probs, times], headers='firstrow', tablefmt='latex_raw'), file=prob_tex_file)
                tex_trail = str(trail).replace('_', '\\_')
                print(f"\\caption{{Count probability results for {relevant_rounds} of {tex_trail}}}", file=prob_tex_file)
                print("\\end{table}", file=prob_tex_file)
                print("", file=prob_tex_file)


    if count_tweakey_results:
        print(f'# Count Tweakey Results', file=md_file)
        print(file=md_file)
        print(tabulate.tabulate(count_tweakey_results, headers='keys', tablefmt='github') + '\n', file=md_file)
        print(file=md_file)

    if count_tweakey_lin_results:
        print(f'# Count Tweakey Lin Results', file=md_file)
        print(file=md_file)
        print(tabulate.tabulate(count_tweakey_lin_results, headers='keys', tablefmt='github') + '\n', file=md_file)
        print(file=md_file)

    if lin_constraints:
        print(f'## Linear Constraints', file=md_file)
        print(file=md_file)
        print(tabulate.tabulate(lin_constraints, headers='keys', tablefmt='github') + '\n', file=md_file)
        print(file=md_file)

    if count_tweakey_sat_results:
        tweakey_sat_conditions = []

        for k, v in count_tweakey_sat_results.items():
            cipher, trail, kind = k

            if len(v['tweakey conditions']) > 0:
                v['tweakey conditions'] = sorted(v['tweakey conditions'])
                it = iter(v['tweakey conditions'])
                first = next(it)
                tweakey_sat_conditions.append({'cipher': cipher, 'trail': v['trail'], 'kind': v['kind'], 'condition': first})
                for cond in it:
                    tweakey_sat_conditions.append({'cipher': '', 'trail': '', 'kind': '', 'condition': cond})
                tweakey_sat_conditions.append({'cipher': '', 'trail': '', 'kind': '', 'condition': ''})
            v['tweakey conditions'] = len(v['tweakey conditions'])
            v['time'] = format_time(v['time'])

            for confidence in [0.8, 0.95, 0.99]:
                lower, upper = wilson_score_interval(v['count_sat'], v['trials'], confidence)
                # lower = 0 if lower < 0 else lower
                if lower > 0 and upper > 0:
                    fmted_ci = f'2^({log2(lower):.2f}..{log2(upper):.2f})'
                else:
                    fmted_ci = f'{fmt_log2(lower)}..{fmt_log2(upper)}'
                count_tweakey_sat_results[k][f'{confidence*100:.0f}% CI'] = fmted_ci

                if confidence == 0.8:
                    # tweakey_size = v['_tweakey_size']
                    lower_log2 = log2(lower) if lower > 0 else float('-inf') if lower == 0 else float('nan')
                    upper_log2 = log2(upper) if upper > 0 else float('-inf') if upper == 0 else float('nan')
                    latex_table[(trail, kind)][KEY_COUNT_SAT] = fmt_ci_latex_log2(lower_log2, upper_log2)
            del v['_tweakey_size']


        print(f'# Count Tweakey SAT results', file=md_file)
        print(file=md_file)
        print(tabulate.tabulate(count_tweakey_sat_results.values(), headers='keys', tablefmt='github') + '\n', file=md_file)
        print(file=md_file)

        print(f'## Constraints', file=md_file)
        print(tabulate.tabulate(tweakey_sat_conditions, headers='keys', tablefmt='github') + '\n', file=md_file)


        latex_table_list = [{'trail': trail, 'kind': kind, **v} for ((trail, kind), v) in latex_table.items()]
        latex_table_list = sorted(latex_table_list, key=lambda x: (x['trail']))
        tex_file.write(tabulate.tabulate(latex_table_list, headers='keys', tablefmt='latex_raw') + '\n')



if __name__ == '__main__':
    with open('results.md', 'w') as md_file, open('results.tex', 'w') as tex_file:
        with open('results_prob.md', 'w') as prob_file, open('results_prob.tex', 'w') as prob_tex_file:
            ret = gather_results(sys.argv[1:], md_file, tex_file, prob_file, prob_tex_file)
    raise SystemExit(ret)
