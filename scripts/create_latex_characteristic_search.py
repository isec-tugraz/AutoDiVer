import os
import numpy as np

directory = "./found_trails"

_PREAMBLE = r"""
\begin{table}[htp!]
  \centering
  \caption{Caption.}\label{tab:results}
  \setlength{\tabcolsep}{8pt}
  \begin{tabular}{lrrrr}
    \toprule
    Cipher                       & \#R & EDP & Search Time & Figure Reference \\
    \midrule
""".strip("\n")

_TABLE_END = r"""
    \bottomrule
  \end{tabular}
\end{table}
""".strip("\n")


#print(_PREAMBLE)

for filename in sorted(os.listdir(directory)):

    file_path = os.path.join(directory, filename)

    with np.load(file_path, allow_pickle=True) as data:
        cipher_name = data["cipher_name"]
        round_number = data["num_rounds"]  # printed as round_number
        log_probability = data["log_probability"]
        search_time = data["search_time"]

        print(f"{cipher_name}, {round_number},{log_probability}, {search_time:.3f}s")


#print(_TABLE_END)
