#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: sources = skinny/skinny-c/src/skinny128-cipher.c
#distutils: include_dirs = skinny/skinny-c/include/
cimport cython
from libc.stdio cimport printf
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint32_t, uint64_t
# from skinny cimport *
cdef int _skinny_enc_ecb(uint8_t *ct, const uint8_t *pt, const uint8_t tweakey[48], unsigned int ct_len, unsigned int numrounds) noexcept nogil:
    if numrounds > 56 or ct_len % 16 != 0:
        return 0
    cdef Skinny128Key_t _key;
    skinny128_set_key(&_key, &tweakey[0], 48)
    _key.rounds = numrounds
    for i in range(0, ct_len, 16):
        skinny128_ecb_encrypt(&ct[i], &pt[i], &_key)
    return 1
def skinny_enc_ecb(const uint8_t[::1] pt not None, const uint8_t[::1] tweakey not None, unsigned int numrounds) -> uint8_t[:]:
    ct = bytearray(pt.shape[0])
    cdef uint8_t[::1] ct_view = ct
    if not _skinny_enc_ecb(&ct_view[0], &pt[0], &tweakey[0], pt.shape[0], numrounds):
        raise ValueError('invalid parameter')
    return bytes(ct)
cdef int _romulush_reduce(uint8_t result[32], const uint8_t lr[32], const uint8_t msg[32], int numrounds) noexcept nogil:
    cdef uint8_t pt[32];
    memcpy(&pt[0], &lr[0], 16)
    memcpy(&pt[16], &lr[0], 16)
    pt[16] ^= 1
    cdef uint8_t tweakey[48]
    memcpy(&tweakey[0], &lr[16], 16)
    memcpy(&tweakey[16], &msg[0], 32)
    if not _skinny_enc_ecb(result, pt, tweakey, 32, numrounds):
        return 0
    for i in range(32):
        result[i] ^= pt[i]
    return 1
def romulush_reduce(const uint8_t[::1] lr, const uint8_t[::1] msg, int numrounds):
    if lr.shape[0] != 32 or msg.shape[0] != 32:
        raise ValueError()
    result = bytearray(32)
    cdef uint8_t[::1] result_view = result
    _romulush_reduce(&result_view[0], &lr[0], &msg[0], numrounds)
    return bytes(result)
cdef void _romulush(uint8_t hash[32], const uint8_t *msg, size_t msg_len, int numrounds) noexcept nogil:
    cdef uint8_t buf[32]
    cdef ssize_t i = 0
    memset(hash, 0, 32)
    while i < <ssize_t> msg_len - 31:
        _romulush_reduce(hash, hash, &msg[i], numrounds)
        i += 32
    memset(&buf[0], 0, 32)
    cdef ssize_t rem_len = msg_len - i
    memcpy(buf, &msg[i], rem_len)
    buf[31] = rem_len
    hash[0] ^= 2
    _romulush_reduce(hash, hash, &buf[0], numrounds)
def romulush(const uint8_t[::1] msg, int numrounds=40):
    if numrounds > 56:
        raise ValueError('cannot do more than 56 more')
    hash = bytearray(32)
    cdef uint8_t[::1] hash_view = hash
    _romulush(&hash_view[0], &msg[0], msg.shape[0], numrounds)
    return bytes(hash)