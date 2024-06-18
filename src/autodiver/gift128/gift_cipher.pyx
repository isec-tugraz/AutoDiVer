#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: sources = src/autodiver/gift128/gift128_ref.c
cimport cython
from libc.stdio cimport printf
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t
import numpy as np
cdef extern from *:
    """
    void enc(unsigned char *pt, const unsigned char *key, int num_rounds);
    """
    void enc(unsigned char *pt, const unsigned char *key, int num_rounds) nogil;
def gift128_enc(const uint8_t[::1] pt not None, const uint8_t[::1] key not None, unsigned int numrounds=40) -> uint8_t[:]:
    cdef ssize_t i
    if pt.shape[0] != 32 or key.shape[0] != 32:
        raise ValueError('invalid parameter (pass in arrays of nibbles)')
    for i in range(16):
        if pt[i] > 0xf:
            raise ValueError('invalid parameter (pass in arrays of nibbles)')
    for i in range(32):
        if key[i] > 0xf:
            raise ValueError('invalid parameter (pass in arrays of nibbles)')
    ct = bytearray(pt)
    cdef uint8_t[::1] ct_view = ct
    enc(&ct_view[0], &key[0], numrounds)
    return np.array(ct)