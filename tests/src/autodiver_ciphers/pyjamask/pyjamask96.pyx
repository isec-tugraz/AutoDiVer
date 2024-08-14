# SPDX-License-Identifier: GPL-3.0-or-newer

#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: sources = src/autodiver_ciphers/pyjamask/pyjamask-c/pyjamask.c
#distutils: include_dirs = src/autodiver_ciphers/pyjamask/pyjamask-c/
cimport cython

from libc.stdint cimport uint8_t, uint32_t

import numpy as np

cdef extern from *:
    """
    void pyjamask_96_enc(const uint8_t *plaintext, const uint8_t *key, uint8_t *ciphertext, int num_rounds_ks, int num_rounds);
    void pyjamask_96_enc_longkey(const uint8_t *plaintext, const uint32_t *long_key, uint8_t *ciphertext, int num_rounds);
    """

    void pyjamask_96_enc(const uint8_t *plaintext, const uint8_t *key, uint8_t *ciphertext,  int num_rounds_ks, int num_rounds) nogil;
    void pyjamask_96_enc_longkey(const uint8_t *plaintext, const uint32_t *long_key, uint8_t *ciphertext, int num_rounds) nogil; # for testing longkey version



def pypyjamask_96_enc(const uint8_t[::1] pt not None, const uint8_t[::1] key not None, int numrounds_ks=14, int numrounds=14) -> uint8_t[:]:
    if pt.shape[0] != 3 * 4:
        raise ValueError('invalid plaintext size: 3*32bit = 96bit required')
    if key.shape[0] != 4*4:
        raise ValueError('invalid key size: 128bit required')
    ct = bytearray(pt)
    cdef uint8_t* ct_view = ct
    pyjamask_96_enc(<uint8_t*>&pt[0], <uint8_t*>&key[0], ct_view, numrounds_ks, numrounds)
    return bytearray(ct)

def pypyjamask_96_enc_longkey(const uint8_t[::1] pt not None, const uint8_t[::1] long_key not None, int numrounds=14) -> uint8_t[:]:
    # assertions? see gift
    if pt.shape[0] != 3*4:
        raise ValueError('invalid plaintext size: 3*32bit = 96bit required')
    if long_key.shape[0] != 4*4*(numrounds + 1):
        print(long_key.shape[0])
        raise ValueError('invalid state size: 3*32*numrounds bit required')
    ct = bytearray(pt)
    cdef uint8_t* ct_view = ct
    pyjamask_96_enc_longkey(<uint8_t*>&pt[0], <uint32_t*>&long_key[0], ct_view, numrounds)
    return bytearray(ct)
