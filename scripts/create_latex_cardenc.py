from pathlib import Path
from io import StringIO
import os
import numpy as np

directory = "./misc/card_enc_var_round_fixed_seed"

_PREAMBLE = r"""
\begin{table}[htbp]
\centering
\setlength{\tabcolsep}{2pt}
\begin{tabular}{|c|c|c|c|c|c|c|c|}
  \hline
  \diagbox{\faCheck}{\faClose} & Bound & Card. Net & kmTotalizer & mTotalizer & Seq. Ctr & Sorting Net & Totalizer \\
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
            print(f"{search_time * 1000:.0f}\,ms", file=time, end="")
    elif search_time < 1:
        print(f"{search_time:.1f}\,s", file=time, end="")
    elif search_time <= 60:
        print(f"{search_time:.0f}\,s", file=time, end="")
    elif search_time <= 60 * 60:
        print(f"{search_time / 60:.0f}\,m", file=time, end="")
    elif search_time <= 60 * 60 * 24:
        print(f"{search_time / (60 * 60):.0f}\,h", file=time, end="")
    else:
        print(f"{search_time / (60 * 60 * 24):.0f}\,d", file=time, end="")

    return time


def main():
    tex_file = Path.cwd() / "../2026_project_juettler/thesis/chapters/cardenc_table.tex"
    result = StringIO()

    print(_PREAMBLE, file=result)

    for cipher in sorted(os.listdir(directory)):
        print(f""" \\cipher{{{map_cipher_name[cipher]}-""", file=result, end="")

        subdirectory = os.path.join(directory, cipher)
        files = sorted(os.listdir(subdirectory))

        # create preamble with num_rounds for this char and boundary that was used
        char = files[0]
        file_path = os.path.join(subdirectory, char)
        with np.load(file_path, allow_pickle=True) as data:
            rounds = data["num_rounds"]
            boundary = data["boundary"]
            print(f"""R{rounds}}} """, file=result, end="")
            print(f"& \diagbox{{{boundary}}}{{{boundary - 1}}} ", file=result, end="")


        # find fastest cardencs
        fastest_sat = 0xFFFFFFFF
        fastest_unsat = 0xFFFFFFFF
        slowest = 0.0
        for char in files:

            file_path = os.path.join(subdirectory, char)
            with np.load(file_path, allow_pickle=True) as data:
                time_sat = data["stat_sat_search"][0]
                time_unsat = data["stat_unsat_search"][0]

                if time_sat > slowest:
                    print(time_sat, char, "sat", cipher)
                    slowest = time_sat

                if time_unsat > slowest:
                    print(time_unsat, char, "unsat", cipher)
                    slowest = time_unsat

                if time_sat < fastest_sat:
                    print(time_sat, char, "sat", cipher)
                    fastest_sat = time_sat

                if time_unsat < fastest_unsat:
                    print(f"{time_unsat} < {fastest_unsat}")
                    print(time_unsat, char, "unsat", cipher)
                    fastest_unsat = time_unsat

        for char in files:
            print(char)
            file_path = os.path.join(subdirectory, char)
            with np.load(file_path, allow_pickle=True) as data:
                time_sat = format_time(data["stat_sat_search"][0])
                time_unsat = format_time(data["stat_unsat_search"][0])

                print(time_sat.getvalue(), time_unsat.getvalue())

                print(f"& \diagbox", file=result, end="")
                if data["stat_sat_search"][0] == fastest_sat:
                    print(f"{{\\textcolor{{teal}}{{{time_sat.getvalue()}}}}}", file=result, end="")
                elif data["stat_sat_search"][0] == slowest:
                    print(f"{{\\textcolor{{tug}}{{{time_sat.getvalue()}}}}}", file=result, end="")
                else:
                    print(f"{{{time_sat.getvalue()}}}", file=result, end="")

                if data["stat_unsat_search"][0] == fastest_unsat:
                    print(f"{{\\textcolor{{teal}}{{{time_unsat.getvalue()}}}}}", file=result, end="")
                elif data["stat_unsat_search"][0] == slowest:
                    print(f"{{\\textcolor{{tug}}{{{time_unsat.getvalue()}}}}}", file=result, end="")
                else:
                    print(f"{{{time_unsat.getvalue()}}}", file=result, end="")

                # print(f"& \diagbox{{{time_sat.getvalue()}}}{{{time_unsat.getvalue()}}} ", file=result, end="")

        # print(" & & & & & ", file=result, end="")
        print("\\\\ \n \hline", file=result)

    print("""\end{tabular}
\label{tab:card_enc}
\caption{Runtime comparison of cardinality encodings of models for different ciphers with the given bound. Satisfiable instances are marked with \\faCheck, unsatisfiable instances with \\faClose. The fastest satisfiable and unsatisfiable problem instances are marked in \\textcolor{teal}{green}, while the slowest instance per cipher is marked in \\textcolor{tug}{red}. All encodings are from PySAT~\cite{sat/IgnatievMM18}. Results were acquired on an Intel(R) Xeon(R) E5-2699 CPU @ 2.20GHz.}
\end{table}""", file=result)

    tex_file.write_text(result.getvalue())

if __name__ == "__main__":
    main()