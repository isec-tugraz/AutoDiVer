#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
cimport cython
from libc.stdlib cimport rand, srand
from libc.math cimport log2
from libc.string cimport memset, memcpy
from libc.stdio cimport printf
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t
from cython.view cimport array as cvarray
# cimport numpy as np
from typing import *
from skinny.skinny cimport *
from collections import Counter
import numpy as np
from os import urandom
cdef uint8_t sbox[256];
sbox[:] = [
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
]
def randomize():
    srand(int.from_bytes(urandom(4), 'big') & 0x7fffffff)
randomize()
cdef void fill_ddt4(const uint8_t[:] sbox, uint16_t[:, :, :, :] ddt4) noexcept nogil:
    if not (sbox.shape[0] == ddt4.shape[0] == ddt4.shape[1] == ddt4.shape[2] == ddt4.shape[3]):
        return
    cdef int l = sbox.shape[0]
    cdef uint8_t dout1
    cdef uint8_t dout2
    cdef uint8_t dout12
    for val in range(l):
        for d1 in range(l):
            for d2 in range(l):
                dout1 = sbox[val] ^ sbox[val ^ d1]
                dout12 = sbox[val ^ d2] ^ sbox[val ^ d1 ^ d2]
                dout2 = sbox[val] ^ sbox[val ^ d2]
                if dout12 == dout1:
                    ddt4[d1][d2][dout1][dout2] += 1
cdef void fill_ddt3(uint16_t[:, :, :] ddt3, const ssize_t[:] index0, const ssize_t[:] index1, const ssize_t[:] index2, const uint16_t[:] values):
    if not (ddt3.shape[0] == ddt3.shape[1] == ddt3.shape[2]):
        return
    if not (index0.shape[0] == index1.shape[0] == index2.shape[0] == values.shape[0]):
        return
    cdef ssize_t l = index0.shape[0]
    for i in range(l):
        ddt3[index0[i], index1[i], index2[i]] += values[i]
class DDT4:
    @cython.boundscheck(True)
    def __init__(self, const uint8_t[:] sbox, cache_file: Optional[str]):
        self.ddt4 = np.zeros((len(sbox),) * 4, dtype=np.uint16)
        self.sbox = sbox
        try:
            cache = np.load(cache_file)
            self.nz_indices = tuple(cache.get('nz_indices'))
            self.nz_entries = cache.get('nz_entries')
            self.ddt4[self.nz_indices] = self.nz_entries
            self.ddt3 = self._build_ddt3()
            return
        except IOError:
            pass
        fill_ddt4(sbox, self.ddt4)
        self.nz_indices = np.where(self.ddt4 != 0)
        self.nz_entries = self.ddt4[self.nz_indices]
        self.ddt3 = self._build_ddt3()
        np.savez(cache_file, nz_indices=self.nz_indices, nz_entries=self.nz_entries)
    def _build_ddt3(self):
        ddt3 = np.zeros((len(self.sbox), ) * 3, dtype=np.uint16)
        fill_ddt3(ddt3, self.nz_indices[0], self.nz_indices[1], self.nz_indices[2], self.nz_entries)
        return ddt3
    def __getitem__(self, *args, **kwargs):
        return self.ddt4.__getitem__(*args, **kwargs)
    def __setitem__(self, *args, **kwargs):
        return self.ddt4.__setitem__(*args, **kwargs)
    def spectrum(self) -> Counter:
        return Counter(self.nz_entries)
cdef int fill_conditional_ddt(const uint8_t[:] sbox, uint16_t[:, :, :] conditional_ddt) nogil:
    if not (sbox.shape[0] == conditional_ddt.shape[0] == conditional_ddt.shape[1] == conditional_ddt.shape[2]):
        return False
    cdef int l = sbox.shape[0]
    cdef uint8_t dout
    for key in range(l):
        for x in range(l):
            for din in range(l):
                dout = sbox[key ^ sbox[x]] ^ sbox[key ^ sbox[x ^ din]]
                conditional_ddt[key][din][dout] += 1
    return True
def get_conditional_ddt(const uint8_t[:] sbox) -> np.ndarray:
    result = np.zeros((sbox.shape[0], sbox.shape[0], sbox.shape[0]), dtype=np.uint16)
    if not fill_conditional_ddt(sbox, result):
        raise RuntimeError()
    return result
def get_column_probability(
        const uint8_t[:] x2_vals not None,
        const uint8_t[:] x3_vals not None,
        const uint8_t[:] y0_vals not None,
        const uint8_t[:] y1_vals not None,
        #const uint8_t[:] y2_vals not None,
        const uint8_t[:] y3_vals not None,
        bint add_round_constant,
        ) -> Tuple[float, float, float, float]:
    """
    Calculates probabilty of the differential transition in parts of one column.
    The probaility of the transition `x0, x2, x3 --> y0, y1, y3` is calculated
    based on the supplied solution set of `x` and `y`.
    """
    cdef bint y0_lut[256];
    cdef bint y1_lut[256];
    cdef bint y3_lut[256];
    cdef uint8_t x0_a, x2_a, x3_a, x0_b, x2_b, x3_b
    cdef double p013, p2
    cdef int l2 = x2_vals.shape[0], l3 = x3_vals.shape[0]
    cdef uint8_t rc = 2 if add_round_constant else 0
    cdef int good = 0, total = 256 * l2 * l3
    with nogil:
        for i in range(256):
            y0_lut[i] = False
            y1_lut[i] = False
            y3_lut[i] = False
        for i in range(y0_vals.shape[0]):
            y0_lut[y0_vals[i]] = True
        for i in range(y1_vals.shape[0]):
            y1_lut[y1_vals[i]] = True
        for i in range(y3_vals.shape[0]):
            y3_lut[y3_vals[i]] = True
        for i0 in range(256):
            x0_a = i0;
            if not y1_lut[x0_a]:
                continue
            for i2 in range(l2):
                x2_a = x2_vals[i2] ^ rc
                if not y3_lut[x0_a ^ x2_a]:
                    continue
                for i3 in range(l3):
                    x3_a = x3_vals[i3]
                    #y0 = x0 ^      x2 ^ x3
                    #y1 = x0
                    #y2 =      x1 ^ x2
                    #y3 = x0 ^      x2
                    good += y0_lut[x0_a ^ x2_a ^ x3_a]
        p013 = <double> good / <double> total
        #p2 = <double> y2_vals.shape[0] / 256
    return p013
cdef void _sbox(uint8_t[:, ::1] state) noexcept nogil:
    cdef uint8_t *state_ptr = &state[0,0]
    for i in range(16):
        state_ptr[i] = sbox[state_ptr[i]]
cdef void _shift_rows(uint8_t[:, ::1] state) noexcept nogil:
    cdef uint8_t tmp_row[4]
    cdef int row, col
    for row in range(1, 4):
        for col in range(4):
            tmp_row[col] = state[row, col]
        for col in range(4):
            state[row, (col + row) % 4] = tmp_row[col]
cdef void _randomize_chaining_value(uint8_t *prefix, uint8_t *lr, int numrounds) noexcept nogil:
    for i in range(32):
        prefix[i] = ord('A') + rand() % 26
    memset(lr, 0, 32)
    _romulush_reduce(lr, lr, prefix, numrounds)
cdef void _build_lut(
        uint8_t[4][4][256] lut,
        const uint8_t[:,:] sbox_in,
        const uint8_t[:,:] sbox_out,
        const int64_t[:,:] tbox_in,
        const int64_t[:,:] tbox_out,
        const int64_t[:,:] unknown) noexcept nogil:
    ...
    cdef uint8_t sbi, sbo, tbi, tbo, unkn, val
    cdef int val_iter
    #memset(&lut, 0, sizeof(lut))
    for row in range(4):
        for col in range(4):
            sbi = sbox_in[row, col]
            sbo = sbox_out[row, col]
            tbi = tbox_in[row, col]
            tbo = tbox_out[row, col]
            unkn = unknown[row, col]
            for val_iter in range(256):
                val = val_iter
                if unkn:
                    lut[row][col][val] = sbox[val] ^ sbox[val ^ sbi] == sbo
                else:
                    lut[row][col][val] = (sbox[val] ^ sbox[val ^ sbi] == sbo and
                                          sbox[val] ^ sbox[val ^ tbi] == tbo and
                                          sbox[val ^ sbi] ^ sbox[val ^ sbi ^ tbi] == tbo)
def build_lut(
        const uint8_t[:,:] sbox_in,
        const uint8_t[:,:] sbox_out,
        const int64_t[:,:] tbox_in,
        const int64_t[:,:] tbox_out,
        const int64_t[:,:] unknown):
    cdef uint8_t[4][4][256] lut,
    _build_lut(lut, sbox_in, sbox_out, tbox_in, tbox_out, unknown)
    result = np.zeros((4,4,256), dtype=int)
    cdef int row, col, val
    for row in range(4):
        for col in range(4):
            for val in range(256):
                result[row, col, val] = lut[row][col][val]
    return result
cdef int32_t _generate_prefix(
        uint8_t *prefix,
        uint8_t *lr,
        const uint8_t[:,:,:] sbox_in,
        const uint8_t[:,:,:] sbox_out,
        const int64_t[:,:,:] tbox_in,
        const int64_t[:,:,:] tbox_out,
        const int64_t[:,:,:] unknown) nogil:
    ...
    cdef int numrounds = sbox_in.shape[0]
    #printf("numrounds: %d\n", numrounds)
    cdef uint8_t pt_buf[4][4]
    cdef uint8_t[:,::1] pt_arr = None
    cdef uint8_t y0, y1, y3, key
    cdef int key_iter
    cdef uint8_t lut[4][4][256]
    _build_lut(lut, sbox_in[1], sbox_out[1], tbox_in[1], tbox_out[1], unknown[1])
    cdef bint conflict
    with gil:
        pt_arr = pt_buf
    cdef int32_t i = 0
    while True:
        i += 1
        _randomize_chaining_value(prefix, lr, numrounds)
        memcpy(&pt_arr[0,0], lr, 16)
        if not unknown[0,0,0]:
            if sbox[pt_arr[0,0]] ^ sbox[pt_arr[0,0] ^ 1] != tbox_out[0,0,0]:
                continue
        _sbox(pt_arr)
        pt_arr[2,0] ^= 2 # round constant
        _shift_rows(pt_arr)
        conflict = False
        for col in range(4):
            # we leave out row 0 as we iterate over all values there anyway
            y0 = pt_arr[2, col] ^ pt_arr[3, col]
            y1 = 0
            y3 = pt_arr[2, col]
            for key_iter in range(256):
                key = key_iter
                if lut[0][col][y0 ^ key] != 0 and lut[1][col][y1 ^ key] != 0 and lut[3][col][y3 ^ key] != 0:
                    break
            else:
                conflict = True
                break
        if not conflict:
            return i
def generate_prefix(
        const uint8_t[:,:,:] sbox_in,
        const uint8_t[:,:,:] sbox_out,
        const int64_t[:,:,:] tbox_in,
        const int64_t[:,:,:] tbox_out,
        const int64_t[:,:,:] unknown):
    """
    returns a prefix such that the characteristic in the first two rounds can
    be satisfied by at least one key.
    """
    prefix = bytearray(32)
    lr = bytearray(32)
    cdef uint8_t[:] prefix_view = prefix
    cdef uint8_t[:] lr_view = lr
    _generate_prefix(&prefix_view[0], &lr_view[0], sbox_in, sbox_out, tbox_in, tbox_out, unknown)
    return bytes(prefix), bytes(lr)
def measure_prefix(
        const uint8_t[:,:,:] sbox_in,
        const uint8_t[:,:,:] sbox_out,
        const int64_t[:,:,:] tbox_in,
        const int64_t[:,:,:] tbox_out,
        const int64_t[:,:,:] unknown,
        ssize_t count) -> np.ndarray:
    """
    executes generate_prefix `count` times and returns a numpy array with
    `count` elements corresponding to the number of tries needed for finding
    each prefix.
    """
    cdef ssize_t i
    cdef int32_t [:]res_view
    res = np.empty(count, np.int32)
    res_view = res
    prefix = bytearray(32)
    lr = bytearray(32)
    cdef uint8_t[:] prefix_view = prefix
    cdef uint8_t[:] lr_view = lr
    for i in range(count):
        res_view[i] = _generate_prefix(&prefix_view[0], &lr_view[0], sbox_in, sbox_out, tbox_in, tbox_out, unknown)
    return res