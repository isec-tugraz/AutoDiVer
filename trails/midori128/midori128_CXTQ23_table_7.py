"""
SAT-Aided Differential Cryptanalysis of Lightweight Block Ciphers Midori, MANTIS and QARMA
Yaxin Cui, Hong Xu(B), Lin Tan, and Wenfeng Qi
https://doi.org/10.1007/978-981-99-7356-9_1
Table 7. The 10-round differential characteristic of probability 2^-115 for Midori-128
"""
from pathlib import Path
import numpy as np
sr_mapping = np.array([0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8])
sri_mapping = np.array([0, 7, 14, 9, 5, 2, 11, 12, 15, 8, 1, 6, 10, 13, 4, 3])
sr_mapping = sr_mapping.reshape(4, 4).T
sri_mapping = sri_mapping.reshape(4, 4).T
mixing_mat = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],])
def do_shift_rows(state):
    assert state.shape[:2] == (4, 4)
    return state.swapaxes(0, 1).reshape(16, *state.shape[2:])[sr_mapping]
def do_shift_rows_inv(state):
    assert state.shape[:2] == (4, 4)
    return state.swapaxes(0, 1).reshape(16, *state.shape[2:])[sri_mapping]
def do_mix_columns(state):
    assert state.shape == (4, 4)
    out_state = np.zeros((4, 4), dtype=np.uint8)
    for c in range(4):
        col_in = state[:, c]
        col_out = out_state[:, c]
        for r in range(4):
            col_in_sel = col_in[mixing_mat[r] != 0]
            col_out[r] = np.bitwise_xor.reduce(col_in_sel)
    return out_state
if __name__ == '__main__':
    sbox_in = np.array(bytearray.fromhex(
        '00840484010001103001000100000000'
        '00000020000080000080000000000000'
        '00000000000001000000000000000000'
        '00000000000080000000800000008000'
        '04020000000201000400010004020100'
        '00018004004180000040000401010000'
        '02800020028008000000000002000920'
        '00010000400000000000000000000400'
        '00800000000000000000000000000000'
        '00800000000000000080000000800000'
        '00a001000000010400a0010400a00004'
    )).reshape(-1, 4, 4)
    sbox_out = np.array([
        do_shift_rows_inv(do_mix_columns(sbox_in[i]))
        for i in range(1, len(sbox_in))
    ])
    sbox_in = sbox_in[:-1]
    assert np.all((sbox_in == 0) == (sbox_out == 0))
    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, sbox_in=sbox_in, sbox_out=sbox_out)