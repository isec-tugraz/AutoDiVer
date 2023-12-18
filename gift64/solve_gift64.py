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
class SOLVE_MODEL(CIPHER_MODEL):
    def __init__(self, cipherName, sboxList, blockSize, sboxSize, nRound, nSbox):
        self._trailFileName = "../trails/{}_{}.txt".format(cipherName,str(nRound))
        trail = []
        with open(self._trailFileName, 'r') as file:
            for line in file:
                hex_values = line.strip().split(',')
                decimal_values = [int(hex_value, 16) for hex_value in hex_values]
                trail.append(decimal_values)
                # print(decimal_values)
        super().__init__(sboxList, blockSize, sboxSize, nRound, nSbox,trail)
        checkenviroment()
        self._cipherName = cipherName
        self._modelFileName = "../model/{}_{}.cnf".format(cipherName,str(self._nRound))
        self._solFileName = "../sol/{}_{}.sol".format(cipherName,str(self._nRound))
        self._modelFile = open(self._modelFileName, "w")
        self._modelFile.write(str(self._completeCnf))
        self._modelFile.close()
        self.solve()
    def solve(self):
        parameters = ["approxmc", self._modelFileName]
        R = subprocess.run(parameters, capture_output=True)
        R = R.stdout.decode()
        R = R.splitlines()
        isSatisfiable = R[len(R) - 2][2:]
        if isSatisfiable == "SATISFIABLE":
            nSolutions = R[len(R) - 3][34:]
            print(isSatisfiable)
            print(nSolutions)
            self._solFile = open(self._solFileName, "w")
            self._solFile.write(isSatisfiable)
            self._solFile.write("\n")
            self._solFile.write(nSolutions)
            self._solFile.close()
        else:
            print(isSatisfiable)
if __name__ == "__main__":
    cipherName = "gift64"
    sboxList = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
    blockSize = 64
    sboxSize  =  4
    rounds    =  7
    nrofSbox  = 16
    gift = SOLVE_MODEL(cipherName, sboxList, blockSize, sboxSize, rounds, nrofSbox)