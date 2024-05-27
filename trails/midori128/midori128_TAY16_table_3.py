"""
Truncated and Multiple Differential Cryptanalysis of Reduced Round Midori128
Mohamed Tolba, Ahmed Abdelkhalek, and Amr M. Youssef
https://doi.org/10.1007/978-3-319-45871-7_1
Table 3. The 2^−123 10-round characteristic of Midori128
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
        "00000000001000000000100000000000"
        "00000000000000000000000000080800"
        "40004040000000000010101000000000"
        "40400040404000000808000808080000"
        "00400040000000400800000008000800"
        "40000040004040000800000800000000"
        "40404000004000400808000808000800"
        "00400000400040000008000008000800"
        "00000000004000000000080000000000"
        "00000000000000000000000000080800"
        "40004040000000000010101000000000"
    )).reshape(-1, 16)[:, ::-1].reshape(-1, 4, 4).swapaxes(-1, -2)
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