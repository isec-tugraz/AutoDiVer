#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: sources = src/autodiver/rectangle128/rectangle_ref.c
cimport cython
from libc.stdio cimport printf
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint64_t
import numpy as np


cdef extern from *:
    """
    uint64_t encrypt(uint64_t msg, uint64_t *orkey, int rounds);
    void nibbleToKey(uint8_t *arr, uint64_t *key);
    uint64_t nibbleToState(uint8_t *arr);
    """
    uint64_t encrypt(uint64_t msg, uint64_t *orkey, int rounds) nogil;
    void nibbleToKey(uint8_t *arr, uint64_t *key) nogil;
    uint64_t nibbleToState(uint8_t *arr) nogil;


def rectangle_enc(uint64_t pt, uint64_t key0, uint64_t key1, int rounds):
    cdef uint64_t key_arr[2]
    cdef uint64_t result
    key_arr[0] = key0
    key_arr[1] = key1
    result = encrypt(pt, key_arr, rounds)
    # print('result:', result)
    return result


def nibble_to_block(arr):
    cdef uint8_t[:] a = arr
    cdef uint64_t state = nibbleToState(&a[0])
    return state


def nibble_to_key(arr):
    cdef uint8_t[:] a = arr
    cdef uint64_t key[2]
    nibbleToKey(&a[0], key)
    
    key_arr = [key[0], key[1]]
    return key_arr
