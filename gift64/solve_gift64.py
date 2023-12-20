import os
import sys
import subprocess
from gift64 import *
from trails import *
def checkenviroment():
    """
    Basic checks if the enviroment is set up correctly
    """
    if not os.path.exists("../model/"):
        os.makedirs("../model/")
    if not os.path.exists("../sol/"):
        os.makedirs("../sol/")
    # if not os.path.exists(PATH_APPROXMC):
    #     print("WARNING: Could not find APPROXMC binary, please check")
    return
class SolveModel(CipherModel):
    def __init__(self, cipherName, sboxList, blockSize, sboxSize, nRound, nSbox):
        self._trailFileName = "../trails/{}_{}.txt".format(cipherName,str(nRound))
        trail = []
        with open(self._trailFileName, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                hex_values = list(line)[:nSbox]
                decimal_values = [int(hex_value, 16) for hex_value in hex_values]
                trail.append(decimal_values)
                # print(decimal_values)
        sbox_in = trail[::2]
        sbox_out = trail[1::2]
        super().__init__(sboxList, blockSize, sboxSize, nRound, nSbox, sbox_in, sbox_out)
        checkenviroment()
        self._cipherName = cipherName
        self._modelFileName = "../model/{}_{}.cnf".format(cipherName,str(self._nRound))
        self._solFileName = "../sol/{}_{}.sol".format(cipherName,str(self._nRound))
        self._modelFile = open(self._modelFileName, "w")
        self._modelFile.write(str(self.cnf))
        self._modelFile.close()
        self.solve()
if __name__ == "__main__":
    cipherName = "gift64"
    sboxList = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
    blockSize = 64
    sboxSize  =  4
    rounds    =  4
    nrofSbox  = 16
    gift = SolveModel(cipherName, sboxList, blockSize, sboxSize, rounds, nrofSbox)