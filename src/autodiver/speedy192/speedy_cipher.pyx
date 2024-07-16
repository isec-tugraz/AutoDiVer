#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: sources = src/autodiver/speedy192/speedy192.c
cimport cython

from libc.stdio cimport printf
from libc.string cimport memcpy, memset

from libc.stdint cimport uint8_t, uint64_t

import numpy as np

cdef extern from *:
    """
    void Encrypt(uint8_t *plaintext, uint8_t *key, int rounds);
    """
    void Encrypt(uint8_t *plaintext, uint8_t *key, int rounds) nogil;

def speedy192_enc(const uint8_t[:] pt not None, const uint8_t[:] key not None, int rounds)-> uint8_t[:]:

    cdef ssize_t i

    ct = bytearray(pt)
    cdef uint8_t[:] ct_view = ct
    Encrypt(&ct_view[0], &key[0], rounds)
    return np.array(ct)
