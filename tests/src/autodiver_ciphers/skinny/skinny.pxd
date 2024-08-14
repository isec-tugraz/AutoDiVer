# SPDX-License-Identifier: MIT

#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: include_dirs = src/autodiver/skinny/skinny-c/include/
cimport cython
from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from "skinny128-cipher.h":
    ctypedef union Skinny128HalfCells_t:
        uint32_t row[2]
        uint64_t lrow
    ctypedef struct Skinny128Key_t:
        unsigned rounds
        Skinny128HalfCells_t schedule[56]


    int skinny128_set_key(Skinny128Key_t *ks, const void *key, unsigned size) nogil
    void skinny128_ecb_encrypt(void *output, const void *input, const Skinny128Key_t *ks) nogil;
    void skinny128_ecb_decrypt(void *output, const void *input, const Skinny128Key_t *ks) nogil;



cdef int _skinny_enc_ecb(uint8_t *ct, const uint8_t *pt, const uint8_t tweakey[48], unsigned int ct_len, unsigned int numrounds) noexcept nogil
cdef int _romulush_reduce(uint8_t *result, const uint8_t *lr, const uint8_t *msg, int numrounds) noexcept nogil
cdef void _romulush(uint8_t hash[32], const uint8_t *msg, size_t msg_len, int numrounds) noexcept nogil
