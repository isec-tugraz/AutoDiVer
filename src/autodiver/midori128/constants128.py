import numpy as np

RC128 = [
    np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 1]
    ]),
    np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]),
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1]
    ]),
    np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
    ]),
    np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]),
    np.array([
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 0],
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 0],
    ]),  # Constant 7
    np.array([
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ]),
    np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
    ]),
    np.array([
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
    ]),
    np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]),
    np.array([
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
    ]),
    np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 0],
    ]),
    np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
    ]),  # Constant 14
    np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 0],
    ]),
    np.array([
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
    ]),
    np.array([
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]),
    np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
    ]),
    np.array([
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
    ])
]

def print_c_format():
    s = "beta[ROUNDS][16] = {\n"
    temp = []
    for c in RC128:
        c_str = [str(c[i, j]) for j in range(4) for i in range(4)]
        c_str = "{" + ", ".join(c_str) + "}"
        temp.append(c_str)
    s = s + ",\n".join(temp) + "};"
    print(s)

if __name__ == '__main__':
    print_c_format()
