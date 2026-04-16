import os
import numpy as np
import re
from pathlib import Path
from io import StringIO
from itertools import groupby


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

prefix_re = re.compile(r'^(.*?_r\d+)')

def group_key(fname):
    m = prefix_re.match(fname)
    return m.group(1) if m else fname

def format_non_power_of_two_ciphers(result: StringIO, files: list, subdirectory: Path):
    for prefix, group in groupby(files, key=group_key):

        # find best char in group when there's multiple:)
        max_prob_group = -256
        best_char_group = ""
        highest_modeled_log_prob = "NONE:)"
        for f in group:
            file_path = os.path.join(subdirectory, f)
            with np.load(file_path, allow_pickle=True) as data:
                log_probability = data["log_probability"]
                modeled_log_prob = data["modeled_log_prob"]
                rounding_mode = data["rounding_mode"]
                if log_probability > max_prob_group:
                    max_prob_group = log_probability
                    best_char_group = file_path
                if data["rounding_mode"] == "up":
                    highest_modeled_log_prob = data["modeled_log_prob"]
            print(file_path, log_probability, modeled_log_prob, rounding_mode)

        with np.load(best_char_group, allow_pickle=True) as data:
            cipher_name = data["cipher_name"]
            num_rounds = data["num_rounds"]
            log_probability = data["log_probability"]
            search_time = data["search_time"]
            modeled_log_prob = data["modeled_log_prob"]

            prob = StringIO()
            if log_probability == 0.0:
                print("1", file=prob, end="")
            elif int(log_probability) == log_probability:
                print(f"2^{{{int(log_probability)}}}", file=prob, end="")
            else:
                print(f"2^{{{log_probability:.2f}}}", file=prob, end="")

            time = StringIO()
            if search_time < 0.1:
                print(f"{search_time * 1000:.0f}ms", file=time, end="")
            elif search_time < 1:
                print(f"{search_time:.1f}s", file=time, end="")
            elif search_time <= 60:
                print(f"{search_time:.0f}s", file=time, end="")
            elif search_time <= 60*60:
                print(f"{search_time/60:.0f}m", file=time, end="")
            elif search_time <= 60*60*24:
                print(f"{search_time/(60*60):.0f}h", file=time, end="")
            else:
                print(f"{search_time / (60 * 60*24):.0f}d", file=time, end="")

            print(f"    {num_rounds} & ${prob.getvalue()}$ & $2^{{-{highest_modeled_log_prob}}}$ & {time.getvalue()} & \\none \\\\", file=result)



directory = "./found_trails"

_TABLE_END = r"""
    \bottomrule
  \end{tabular}
\end{table}
"""

map_cipher_name: dict[str,str] = {
    "ascon" : "Ascon",
    "speck32": "SPECK-32",
    "speck48": "SPECK-48",
    "speck64": "SPECK-64",
    "speck96": "SPECK-96",
    "speck128": "SPECK-128",
    "gift64": "GIFT-64",
    "gift128": "GIFT-128",
    "midori64": "Midori64",
    "midori128": "Midori128",
    "present80": "PRESENT",
    "pyjamask96": "Pyjamask96",
    "rectangle128": "RECTANGLE",
    "skinny64": "SKINNY-64",
    "skinny128": "SKINNY-128",
    "speedy192": "SPEEDY-192",
    "warp": "WARP"
}

non_power_of_two_ciphers = ["gift64", "gift128", "skinny128", "speedy192"]

def main():
    tex_file = Path.cwd() / "../2026_project_juettler/thesis/chapters/results_bigtable.tex"
    result = StringIO()

    for cipher in sorted(os.listdir(directory)):
        if cipher in non_power_of_two_ciphers:
            _PREAMBLE = r"""
\begin{table}[htbp]
\centering
\newcommand{\none}{$-$}
""" + f"\\caption{{Best probabilities for \\cipher{{{map_cipher_name[cipher]}}} by number of rounds.}}\n  \label{{tab:results-{cipher}}}" + r"""\setlength{\tabcolsep}{8pt}
\begin{tabular}{rrrrr}
\toprule
\#R & Found DP & upper bound for DP  & Search Time & Figure Reference \\
\midrule""".strip("\n")
        else:
            _PREAMBLE = r"""
\begin{table}[htbp]
\centering
\newcommand{\none}{$-$}
""" +  f"\\caption{{Best probabilities for \\cipher{{{map_cipher_name[cipher]}}} by number of rounds.}}\n  \label{{tab:results-{cipher}}}" +   r"""\setlength{\tabcolsep}{8pt}
\begin{tabular}{rlrc}
\toprule
\#R & DP & Search Time & Figure Reference \\
\midrule""".strip("\n")

        print(_PREAMBLE, file=result)

        subdirectory = os.path.join(directory, cipher)
        files = sorted([f for f in os.listdir(subdirectory) if f.endswith(".npz")], key=natural_key)
        if cipher in non_power_of_two_ciphers:
            format_non_power_of_two_ciphers(result, files, subdirectory)
        else:
            # assumes there's just one char per round number in repo
            for char in files:
                file_path = os.path.join(subdirectory, char)
                with np.load(file_path, allow_pickle=True) as data:
                    cipher_name = data["cipher_name"]
                    num_rounds = data["num_rounds"]
                    log_probability = data["log_probability"]
                    search_time = data["search_time"]

                    prob = StringIO()
                    if log_probability == 0.0:
                        print("1", file=prob, end="")
                    elif int(log_probability) == log_probability:
                        print(f"2^{{{int(log_probability)}}}", file=prob, end="")
                    else:
                        print(f"2^{{{log_probability:.2f}}}", file=prob, end="")

                    time = StringIO()
                    if search_time < 0.1:
                        print(f"{search_time * 1000:.0f}ms", file=time, end="")
                    elif search_time < 1:
                        print(f"{search_time:.1f}s", file=time, end="")
                    elif search_time <= 60:
                        print(f"{search_time:.0f}s", file=time, end="")
                    elif search_time <= 60 * 60:
                        print(f"{search_time / 60:.0f}m", file=time, end="")
                    elif search_time <= 60 * 60 * 24:
                        print(f"{search_time / (60 * 60):.0f}h", file=time, end="")
                    else:
                        print(f"{search_time / (60 * 60 * 24):.0f}d", file=time, end="")

                print(f"    {num_rounds} & ${prob.getvalue()}$ & {time.getvalue()} & \\none \\\\", file=result)

        print(_TABLE_END, file=result)
        tex_file.write_text(result.getvalue())


if __name__ == "__main__":
    main()
