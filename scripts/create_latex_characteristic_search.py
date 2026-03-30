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


tex_file = Path.cwd() / "../2026_project_juettler/thesis/chapters/results_bigtable.tex"
result = StringIO()

for cipher in sorted(os.listdir(directory)):

    _PREAMBLE = r"""
\begin{table}[htbp]
  \centering
  \newcommand{\none}{$-$}
    """.strip("\n") +  f"\\caption{{Best probabilities for \\cipher{{{map_cipher_name[cipher]}}} by number of rounds.}}\n  \label{{tab:results-{cipher}}}" +   r"""\setlength{\tabcolsep}{8pt}
  \begin{tabular}{rrrr}
  \toprule
  \#R & EDP & Search Time & Figure Reference \\
  \midrule""".strip("\n")
    print(_PREAMBLE, file=result)

    subdirectory = os.path.join(directory, cipher)
    files = sorted([f for f in os.listdir(subdirectory) if f.endswith(".npz")], key=natural_key)
    for prefix, group in groupby(files, key=group_key):

        # find best char in group when there's multiple:)
        max_prob_group = -256
        best_char_group = ""
        for f in group:
            file_path = os.path.join(subdirectory, f)
            with np.load(file_path, allow_pickle=True) as data:
                log_probability = data["log_probability"]
                if log_probability > max_prob_group:
                    max_prob_group = log_probability
                    best_char_group = file_path
            print(file_path, log_probability)

        with np.load(best_char_group, allow_pickle=True) as data:
            cipher_name = data["cipher_name"]
            num_rounds = data["num_rounds"]
            log_probability = data["log_probability"]
            search_time = data["search_time"]

            time = StringIO()


            if search_time < 0.1:
                print(f"{search_time * 1000:.0f}ms", file=time)
            elif search_time <= 1:
                print(f"{search_time:.1f}s", file=time)
            elif search_time <= 60:
                print(f"{search_time:.0f}s", file=time)
            elif search_time <= 60*60:
                print(f"{search_time/60:.0f}m", file=time)
            elif search_time <= 60*60*24:
                print(f"{search_time/(60*60):.0f}h", file=time)
            else:
                print(f"{search_time / (60 * 60*24):.0f}d", file=time)


            print(f"    {num_rounds} & $2^{{{log_probability:.2f}}}$ & {time.getvalue()} & \\none \\\\", file=result)

    print(_TABLE_END, file=result)
    tex_file.write_text(result.getvalue())
