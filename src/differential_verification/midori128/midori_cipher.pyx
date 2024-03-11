#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: sources = src/differential_verification/midori128/midori128.c
cimport cython
from libc.stdio cimport printf
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint64_t
import numpy as np
cdef extern from *:
    """
    void enc_midori128(uint8_t *msg, uint8_t *key, int rounds);
    """
    void enc_midori128(uint8_t *msg, uint8_t *key, int rounds) nogil;
def midori128_enc(const uint8_t[:] pt not None, uint8_t[:] key not None, int rounds)-> uint8_t[:]:
    cdef ssize_t i
    ct = bytearray(pt)
    cdef uint8_t[:] ct_view = ct
    enc_midori128(&ct_view[0], &key[0], rounds)
    return np.array(ct)