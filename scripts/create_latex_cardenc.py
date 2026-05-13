from pathlib import Path
from io import StringIO
import os
import numpy as np

directory = "./misc/card_enc"

_PREAMBLE = r"""
\begin{table}[htbp]
\centering
\begin{tabular}{|c|c|c|c|c|c|c|}
  \hline
  \diagbox{\faCheck}{\faClose} & cardnetwork & kmtotalizer & mtotalizer & seqcounter & sortnetwork & totalizer \\
  \hline
  """.strip("\n")

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

def format_time(search_time: float) -> StringIO:
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

    return time


def main():
    tex_file = Path.cwd() / "../2026_project_juettler/thesis/chapters/cardenc_table.tex"
    result = StringIO()

    print(_PREAMBLE, file=result)

    for cipher in sorted(os.listdir(directory)):
        print(f""" \\cipher{{{map_cipher_name[cipher]}}} """, file=result, end="")

        subdirectory = os.path.join(directory, cipher)
        files = sorted(os.listdir(subdirectory))

        for char in files:
            print(char)
            file_path = os.path.join(subdirectory, char)
            with np.load(file_path, allow_pickle=True) as data:
                time_sat = format_time(data["stat_sat_search"][0])
                time_unsat = format_time(data["stat_unsat_search"][0])

                print(time_sat.getvalue(), time_unsat.getvalue())
                print(f"& \diagbox{{{time_sat.getvalue()}}}{{{time_unsat.getvalue()}}} ", file=result, end="")

        # print(" & & & & & ", file=result, end="")
        print("\\\\ \n \hline", file=result)

    print("""\end{tabular}
\end{table}""", file=result)

    tex_file.write_text(result.getvalue())

if __name__ == "__main__":
    main()