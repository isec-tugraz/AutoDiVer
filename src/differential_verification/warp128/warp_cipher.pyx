#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: sources = src/differential_verification/warp128/warp128.c
cimport cython
from libc.stdio cimport printf
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint64_t
import numpy as np
cdef extern from *:
    """
    void enc(uint8_t *m, uint8_t *k, int rounds);
    """
    void enc(uint8_t *m, uint8_t *k, int rounds) nogil;
def warp_enc(const uint8_t[:] pt not None, uint8_t[:] key not None, int rounds)-> uint8_t[:]:
    cdef ssize_t i
    ct = bytearray(pt)
    cdef uint8_t[:] ct_view = ct
    enc(&ct_view[0], &key[0], rounds)
    return np.array(ct)