import os
import numpy as np

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


for cipher in sorted(os.listdir(directory)):

    _PREAMBLE = r"""
\begin{table}[htbp]
  \centering
  \newcommand{\none}{$-$}
    """.strip("\n") +  f"\\caption{{Best probabilities for \\cipher{{{map_cipher_name[cipher]}}} by number of rounds.}}\n  \label{{tab:results-{cipher}}}" +   r"""\setlength{\tabcolsep}{8pt}
  \begin{tabular}{rrrr}
  \toprule""".strip("\n")
    print(_PREAMBLE)

    subdirectory = os.path.join(directory, cipher)
    for char in sorted(os.listdir(subdirectory)):
        file_path = os.path.join(subdirectory, char)

        with np.load(file_path, allow_pickle=True) as data:
            cipher_name = data["cipher_name"]
            round_number = data["num_rounds"]  # printed as round_number
            log_probability = data["log_probability"]
            search_time = data["search_time"]

            print(f"    {round_number} & $2^{{{log_probability:.2f}}}$ & {search_time:.1f}s & \\none \\\\")

    print(_TABLE_END)
