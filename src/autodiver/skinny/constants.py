from __future__ import annotations

import numpy as np
from pathlib import Path

from ..util import get_ddt


connection_poly_8 = np.array([0] * 9)
connection_poly_8[[0, 2, 8]] = 1

connection_poly_4 = np.array([0] * 5)
connection_poly_4[[0, 1, 4]] = 1

sbox8 = np.array((
  101,  76,   106,  66,   75,   99,   67,   107,  85,   117,  90,   122,  83,   115,  91,   123,
  53,   140,  58,   129,  137,  51,   128,  59,   149,  37,   152,  42,   144,  35,   153,  43,
  229,  204,  232,  193,  201,  224,  192,  233,  213,  245,  216,  248,  208,  240,  217,  249,
  165,  28,   168,  18,   27,   160,  19,   169,  5,    181,  10,   184,  3,    176,  11,   185,
  50,   136,  60,   133,  141,  52,   132,  61,   145,  34,   156,  44,   148,  36,   157,  45,
  98,   74,   108,  69,   77,   100,  68,   109,  82,   114,  92,   124,  84,   116,  93,   125,
  161,  26,   172,  21,   29,   164,  20,   173,  2,    177,  12,   188,  4,    180,  13,   189,
  225,  200,  236,  197,  205,  228,  196,  237,  209,  241,  220,  252,  212,  244,  221,  253,
  54,   142,  56,   130,  139,  48,   131,  57,   150,  38,   154,  40,   147,  32,   155,  41,
  102,  78,   104,  65,   73,   96,   64,   105,  86,   118,  88,   120,  80,   112,  89,   121,
  166,  30,   170,  17,   25,   163,  16,   171,  6,    182,  8,    186,  0,    179,  9,    187,
  230,  206,  234,  194,  203,  227,  195,  235,  214,  246,  218,  250,  211,  243,  219,  251,
  49,   138,  62,   134,  143,  55,   135,  63,   146,  33,   158,  46,   151,  39,   159,  47,
  97,   72,   110,  70,   79,   103,  71,   111,  81,   113,  94,   126,  87,   119,  95,   127,
  162,  24,   174,  22,   31,   167,  23,   175,  1,    178,  14,   190,  7,    183,  15,   191,
  226,  202,  238,  198,  207,  231,  199,  239,  210,  242,  222,  254,  215,  247,  223,  255
), dtype=np.uint8)

sbox4 = np.array([int(x, 16) for x in "c6901a2b385d4e7f"], dtype=np.uint8)

round_constants = np.array([
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
    0x1d, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
    0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04,
    0x09, 0x13, 0x26, 0x0C, 0x19, 0x32, 0x25, 0x0A, 0x15, 0x2A, 0x14, 0x28, 0x10, 0x20
])

sr_mapping = np.array([[ 0,  1,  2,  3],
                       [ 7,  4,  5,  6],
                       [10, 11,  8,  9],
                       [13, 14, 15, 12]])

isr_mapping = np.array([[ 0,  1,  2,  3],
                        [ 5,  6,  7,  4],
                        [10, 11,  8,  9],
                        [15, 12, 13, 14]])

expanded_rc = np.zeros((len(round_constants), 4, 4), np.uint8)
expanded_rc[:, 0, 0] = round_constants & 15
expanded_rc[:, 1, 0] = round_constants >> 4
expanded_rc[:, 2, 0] = 0x2

inv_sbox8 = np.zeros_like(sbox8)
inv_sbox8[sbox8] = np.arange(len(sbox8), dtype=sbox8.dtype)

inv_sbox4 = np.zeros_like(sbox4)
inv_sbox4[sbox4] = np.arange(len(sbox4), dtype=sbox4.dtype)

tweakey_mask = np.array([0xFF] * 8 + [0x00] * 8).reshape(4, 4)
tweakey_perm = np.array([9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7])
inv_tweakey_perm = np.empty_like(tweakey_perm)
inv_tweakey_perm[tweakey_perm] = np.arange(len(tweakey_perm), dtype=tweakey_perm.dtype)



mixing_mat = np.array([
    [1, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 0],
])

inv_mixing_mat = np.array([
    [0, 1, 0, 0],
    [0, 1, 1, 1],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
])


ddt8 = get_ddt(sbox8)


def apply_perm(vec: list, perm: list[int], repeats: int):
    for _ in range(repeats):
        new_vec = [None] * len(vec)
        for i, e in enumerate(perm):
            new_vec[i] = vec[e]
        vec = new_vec
    return vec


def do_inv_shift_rows(state):
    return state.flatten()[isr_mapping]


def do_shift_rows(state):
    return state.flatten()[sr_mapping]


def do_inv_mix_cols(state):
    result = np.zeros_like(state)
    for col in range(4):
        result[:, col] = np.bitwise_xor.reduce(inv_mixing_mat * state[:, col], axis=-1)
    return result


def do_mix_cols(state):
    result = np.zeros_like(state)
    for col in range(4):
        result[:, col] = np.bitwise_xor.reduce(mixing_mat * state[:, col], axis=-1)
    return result


def update_tweakey(tweakeys: list[np.ndarray], block_size: int = 128):
    tweakeys: np.ndarray = np.array(tweakeys, dtype=np.uint8).reshape(3, 16)

    # permute
    tweakeys = tweakeys[:, tweakey_perm]

    # LSFRs to update TK2 and TK3
    i = slice(8)

    if block_size == 128:
        tweakeys[1][i] = (tweakeys[1][i] << 1) + ((tweakeys[1][i] >> 5 & 1) ^ (tweakeys[1][i] >> 7 & 1))
        tweakeys[2][i] = (tweakeys[2][i] >> 1) + (((tweakeys[2][i] & 1) ^ (tweakeys[2][i] >> 6 & 1)) * 128)
    elif block_size == 64:
        tweakeys[1][i] = ((tweakeys[1][i] << 1) & 0xF) + ((tweakeys[1][i] >> 3 & 1) ^ (tweakeys[1][i] >> 2 & 1))
        tweakeys[2][i] = (tweakeys[2][i] >> 1) + (((tweakeys[2][i] & 1) ^ (tweakeys[2][i] >> 3 & 1)) * 8)
    else:
        raise ValueError(f'unsupported block size: {block_size}')


    return tweakeys.reshape(3, 4, 4)



def skinny_verbose(pt: np.ndarray, tk: np.ndarray, numrounds: int):
    states = np.zeros((numrounds + 1, 4, 4), np.uint8)
    tweakeys = np.zeros((numrounds, 3, 4, 4), np.uint8)
    states[0] = pt
    tweakeys[0] = tk

    for r in range(numrounds):
        rc = expanded_rc[r]
        states[r + 1] = do_mix_cols(do_shift_rows(sbox8[states[r]] ^ (np.bitwise_xor.reduce(tweakeys[r], axis=0) & tweakey_mask) ^ rc))
        if r + 1 in range(numrounds):
            tweakeys[r + 1] = update_tweakey(tweakeys[r])
    return states, tweakeys
