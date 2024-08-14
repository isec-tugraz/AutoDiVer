# SPDX-License-Identifier: MIT
#


#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: sources = src/autodiver_ciphers/midori64/midori64.c
cimport cython

from libc.stdio cimport printf
from libc.string cimport memcpy, memset

from libc.stdint cimport uint8_t, uint64_t

import numpy as np

cdef extern from *:
    """
    uint64_t sbox(uint64_t msg);
    uint64_t sr(uint64_t msg);
    uint64_t mc(uint64_t msg);
    uint64_t enc_midori64(uint64_t msg, const uint64_t *key, int rounds);
    uint64_t enc_midori64_longkey(uint64_t msg, const uint64_t *key, int rounds);
    """
    uint64_t sbox(uint64_t msg) noexcept nogil;
    uint64_t sr(uint64_t msg) noexcept nogil;
    uint64_t mc(uint64_t msg) noexcept nogil;
    uint64_t enc_midori64(uint64_t msg, const uint64_t *key, int rounds) noexcept nogil;
    uint64_t enc_midori64_longkey(uint64_t msg, const uint64_t *key, int rounds) noexcept nogil;

def midori64_enc(uint64_t pt, uint64_t key0, uint64_t key1, int rounds):
    cdef uint64_t key_arr[2]
    cdef uint64_t result

    key_arr[0] = key0
    key_arr[1] = key1

    result = enc_midori64(pt, key_arr, rounds)
    # print('result:', result)
    return result

def midori64_sbox(uint64_t msg):
    return sbox(msg)

def midori64_sr(uint64_t msg):
    return sr(msg)

def midori64_mc(uint64_t msg):
    return mc(msg)

def midori64_enc_longkey(uint64_t pt, uint64_t[::1] key, int rounds):
    if key.shape[0] != rounds + 1:
        raise ValueError(f"invalid keysize {key.shape[0]}: rounds + 1 roundkeys required")
    cdef uint64_t result

    result = enc_midori64_longkey(pt, &key[0], rounds)
    # print('result:', result)
    return result
