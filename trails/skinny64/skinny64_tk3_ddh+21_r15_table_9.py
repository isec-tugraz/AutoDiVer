#!/usr/bin/env python3
"""
Efficient Methods to Search for Best Differential Characteristics on SKINNY
Stephanie Delaune, Patrick Derbez, Paul Huynh, Marine Minier, Victor Mollimard, and Charles Prud’homme
Table 9. The Best TK3 differential characteristics on 15 rounds of SKINNY-64
with probability equal to 2^−54
https://doi.org/10.1007/978-3-030-78375-4_8
"""
from pathlib import Path
import numpy as np
def pad(s: bytearray) -> bytearray:
    nibbles = []
    for byte in s:
        nibbles.append(byte >> 4)
        nibbles.append(byte & 0xf)
    return bytearray(nibbles)
if __name__ == '__main__':
    sbox_in = np.array(pad(bytearray.fromhex(
        "0000000140000004"
        "0000000000000020"
        "010D000D0000000D"
        "0020000020000000"
        "0000003000300000"
        "0000C000000C0000"
        "0200000000000200"
        "3000000000000000"
        "0000000000000000"
        "0000000000000000"
        "0010001000000010"
        "0A00000000050000"
        "00000A0000000000"
        "0000000000000000"
        "0000000000040000"
    ))).reshape(15, 4, 4)
    sbox_out = np.array(pad(bytearray.fromhex(
        "0000000820000002"
        "0000000000000010"
        "0A0E000200000002"
        "0030000030000000"
        "000000C000C00000"
        "0000200000020000"
        "0500000000000300"
        "D000000000000000"
        "0000000000000000"
        "0000000000000000"
        "00800090000000A0"
        "0A000000000A0000"
        "00000A0000000000"
        "0000000000000000"
        "0000000000020000"
    ))).reshape(15, 4, 4)
    tweakeys = np.array(pad(bytearray.fromhex(
        "0000080D00000800" "0000040800000500" "00000E0D00000C00"
        "000800000000080D" "000B000000000408" "000E000000000E0D"
        "0D08000000080000" "01090000000B0000" "060F0000000E0000"
        "000000080D080000" "0000000701090000" "0000000F060F0000"
        "D000000800000008" "2000000300000007" "300000070000000F"
        "08000000D0000008" "0F00000020000003" "0700000030000007"
        "08D0000008000000" "064000000F000000" "0B90000007000000"
        "8000000008D00000" "E000000006400000" "B00000000B900000"
        "8000D00080000000" "D0009000E0000000" "50004000B0000000"
        "008000008000D000" "00C00000D0009000" "0050000050004000"
        "008000D000800000" "00A0003000C00000" "00A0002000500000"
        "00008000008000D0" "0000800000A00030" "0000A00000A00020"
        "00008D0000008000" "0000560000008000" "0000D1000000A000"
        "0000008000008D00" "0000001000005600" "000000D00000D100"
        "000D008000000080" "000D00B000000010" "00080060000000D0"
    ))).reshape(15, 3, 4, 4)
    script_file = Path(__file__)
    dst_file = script_file.with_suffix('.npz')
    print(f'Writing to {dst_file}')
    np.savez(dst_file, sbox_in=sbox_in, sbox_out=sbox_out, tweakeys=tweakeys)
    numrounds = len(sbox_in)
    print(script_file.stem)
    for i in range(4, numrounds):
        dst_file = script_file.with_name(script_file.stem + f'_r{i}.npz')
        print(f'Writing to {dst_file}')
        np.savez(dst_file, sbox_in=sbox_in[:i], sbox_out=sbox_out[:i], tweakeys=tweakeys[:i])