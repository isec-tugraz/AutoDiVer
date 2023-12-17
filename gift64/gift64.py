import sys
sys.path.append('../')
from model_util import *
from util import IndexSet
import numpy as np
P64 = np.array((0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3,
                4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7,
                8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,
                12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15))
class CIPHER_MODEL(IndexSet):
    def __init__(self, sboxList, blockSize, sboxSize, nRound, nSbox, trail):
        super().__init__()
        self._sbox = sboxList
        self._sboxSize = sboxSize
        self._blockSize = blockSize
        self._nRound = nRound
        self._nSbox = nSbox
        self._trail = trail #two trails for each sbox layer
        #generate Variables
        self.add_index_array('_sboxIn', (self._nRound+1, self._nSbox, self._sboxSize))
        self.add_index_array('_sboxOut', (self._nRound, self._nSbox, self._sboxSize))
        self._rk = self.keySchedule()
        # print(self._rk)
        # test permutation
        # temp = self.applyPerm(self._sboxIn[0])
        # print(self._sboxIn[0])
        # print(temp)
        self._completeCnf = self.genCnf()
        # print(self._completeCnf)
    def applyPerm(self, array):
        arrayFlat = array.flatten()
        arrayPermuted = arrayFlat[P64]
        arrayOut = arrayPermuted.reshape(16, 4)
        return arrayOut
    def keySchedule(self):
        self.add_index_array('MK', (1, self._nSbox*2, self._sboxSize))
        keyWords = self.MK.flatten()
        keyWords = keyWords.reshape(8, 16)
        # print(f'{keyWords=}')
        RK = []
        for r in range(self._nRound):
            rk = np.empty(len(keyWords[0]) + len(keyWords[1]), dtype=keyWords.dtype)
            rk[0::2] = keyWords[0]
            rk[1::2] = keyWords[1]
            # print(f'{rk=}')
            # print(f'{keyWords[0]=}')
            # print(f'{keyWords[1]=}')
            keyWords[0] = np.roll(keyWords[0], -12)
            keyWords[1] = np.roll(keyWords[1], -2)
            # print(f'{keyWords[0]=}')
            # print(f'{keyWords[1]=}')
            #rotatate the words by 2
            # print(f'{keyWords=}')
            keyWords = np.roll(keyWords, -2, axis=0)
            # print(f'{keyWords=}')
            rk = rk.reshape(16, 2)
            RK.append(rk)
        return RK
    def addKey(self, Y, X, K):
        """
        Y = addKey(X, K)
        """
        CC = CNF()
        for s in range(self._nSbox):
            #Key bits are added in the bit position 0 and 1 of each sbox
            var = [0]
            var.append(Y[s][0])
            var.append(K[s][0])
            var.append(X[s][0])
            CC1 = xorModel(var)
            CC += CC1
            var = [0]
            var.append(Y[s][1])
            var.append(K[s][1])
            var.append(X[s][1])
            CC1 = xorModel(var)
            CC += CC1
            var = [0]
            var.append(Y[s][2])
            var.append(X[s][2])
            CC1 = eqModel(var)
            CC += CC1
            var = [0]
            var.append(Y[s][3])
            var.append(X[s][3])
            CC1 = eqModel(var)
            CC += CC1
        return CC
    def sboxLayer(self, X, Y, inDiff, outDiff):
        """
        Y = S(X)
        """
        CC = CNF()
        for s in range(self._nSbox):
            var = [0]
            var += list(X[s])
            var += list(Y[s])
            # print(f'{var = }')
            CC1 = sboxModel(self._sbox, self._sboxSize, self._sboxSize,\
                    inDiff[s], outDiff[s], var)
            CC += CC1
        return CC
    def genCnf(self):
        cnf = CNF()
        for r in range(0, self._nRound):
            #Sbox Layer
            cnf += self.sboxLayer(self._sboxIn[r], self._sboxOut[r],\
                    self._trail[2*r], self._trail[(2*r)+1])
            #Permutation Layer: no permutation for last round
            if(r != (self._nRound - 1)):
                permOut = self.applyPerm(self._sboxOut[r])
            else:
                permOut = self._sboxOut[r]
            cnf += self.addKey(permOut, self._sboxIn[r+1], self._rk[r])
        return cnf
if __name__ == "__main__":
    cipherName = "GIFT64"
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
    gift = CIPHER_MODEL(cipherName, sbox_list, 64, 4, 1, 16, trail)