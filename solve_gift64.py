import os
import subprocess
from gift64 import *
def checkenviroment():
    """
    Basic checks if the enviroment is set up correctly
    """
    if not os.path.exists("./model/"):
        os.makedirs("./model/")
    if not os.path.exists("./sol/"):
        os.makedirs("./sol/")
    # if not os.path.exists(PATH_APPROXMC):
    #     print("WARNING: Could not find APPROXMC binary, please check")
    return
class SOLVE_MODEL(CIPHER_MODEL):
    def __init__(self, cipherName, sboxList, blockSize, sboxSize, nRound, nSbox, trail):
        super().__init__(sboxList, blockSize, sboxSize, nRound, nSbox,trail)
        checkenviroment()
        self._cipherName = cipherName
        self._modelFileName = "./model/{}_{}.cnf".format(cipherName,str(self._nRound))
        self._solFileName = "./sol/_{}_{}.sol".format(cipherName,str(self._nRound))
        self._modelFile = open(self._modelFileName, "w")
        self._modelFile.write(str(self._completeCnf))
        self._modelFile.close()
        self._solFile = open(self._solFileName, "w")
        self._solFile.close()
        self.solve()
    def solve(self):
        parameters = ["approxmc", self._modelFileName]
        R = subprocess.run(parameters, capture_output=True)
        print(R)
if __name__ == "__main__":
    cipherName = "gift64"
    sbox_list = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
    trail = [
        [0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0xC,0x6,0x0],
        [0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x4,0x2,0x0],
        [0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x6],
        [0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x2],
        [0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x2,0x0,0x0,0x0,0x0],
        [0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x6,0x0,0x0,0x0,0x0],
        [0x0,0x0,0x0,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x2,0x0,0x0,0x0,0x0,0x0],
        [0x0,0x0,0x0,0x0,0x0,0x0,0x7,0x0,0x0,0x0,0x5,0x0,0x0,0x0,0x0,0x0]]
    gift = SOLVE_MODEL(cipherName, sbox_list, 64, 4, 2, 16, trail)