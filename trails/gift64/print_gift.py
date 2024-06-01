#!/usr/bin/env python3
def state2tex(S, nb=1):
    res = ""
    for row in range(4):
        for col in range(4):
            if S[row][col]:
                res += r"\Cell{ss" + str(row) + str(col) + "}{" + hex(S[row][col])[2:].zfill(nb) + "}"
    return res
if __name__ == '__main__':
    char = """
000000c000000060
0000002000000020
0000000000000202
0000000000000505
0000000500000005
0000000200000002
0000000002020000
0000000005050000
0000005000000050
0000002000000020
0000000000000202
0000000000000505
0000000500000005
0000000200000002
0000000002020000
0000000005050000
0000005000000050
0000002000000020
0000000000000202
0000000000000a0a
000a0000000a0000
0001000000010000
0000000000001010
0000000000006080
0004000a00000000
0005000100000000
    """
    charwords = char.split()
    binary = lambda s : f'{int(s,16):0>64b}'
    perm = "0 17 34 51 48 1 18 35 32 49 2 19 16 33 50 3 4 21 38 55 52 5 22 39 36 53 6 23 20 37 54 7 8 25 42 59 56 9 26 43 40 57 10 27 24 41 58 11 12 29 46 63 60 13 30 47 44 61 14 31 28 45 62 15".split()
    print(r"""
    \smallgifttrue
    \giftinit[i]
    \spnlinktrue
""")
    for r in range(len(charwords)//2):
        si = list(reversed(binary(charwords[2*r])))
        so = list(reversed(binary(charwords[2*r+1])))
        ibitstr = ", ".join([str(j) for j, sij in enumerate(si) if int(sij)])
        sboxstr = ", ".join([str(s) for s in range(len(si)//4) if '1' in si[4*s:4*s+4]])
        permstr = ", ".join([str(j) + "/" + perm[j] for j, soj in enumerate(so) if int(soj)])
        print(r"""
    \giftround
    \giftmarkbits[diff]{""" + ibitstr + r"""}{""" + sboxstr + r"""}{""" + permstr + r"""}
""")
    print(r"""
    \giftfini
    \foreach \i in {0,4,...,\bits} { \draw (b\i|-here) node[below,gray,inner sep=0pt,font=\tiny] {\i}; }
""")