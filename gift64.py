from sbox_model import *
class CIPHER_MODEL(object):
    def __init__(self, sbox_list, block_size, sbox_size, number_of_rounds, number_of_sbox, trail):
        self._sbox_cnf = sbox_propagation_to_cnf(sbox_list, sbox_size, sbox_size)
        self._sbox_size = sbox_size
        self._block_size = block_size
        self._rounds = number_of_rounds
        self._number_of_sbox = number_of_sbox
        self._trail = trail #list of binary strings of size 2*rounds, two trails for each sbox layer
        self._complete_cnf = CNF()
        #variables_X0 [sbox] variables_Y0 [Linear part] variables_Y0 [key_add variables_K0] variables_X1 ....
        self._variable_counter = 0
        self._variables_X = self.gen_variables(self._rounds+1)
        self._variables_Y = self.gen_variables(self._rounds)
        self._variables_K = self.key_schedule()
        print("Total Variables = ", self._variable_counter)
        # self.print_variables()
        # self.gen_cnf()
    def gen_variables(self, r):
        X = []
        for r in range(r):
            x = []
            for i in range(self._block_size):
                self._variable_counter += 1
                x.append(self._variable_counter)
            X.append(x)
        return X
    def P(self, array_in):
        array = [0 for i in range(0,64)]
        for i in range(0,64):
            array[((i >> 4) << 2) + (((3*((i & 0xf)>>2) + (i & 3)) & 3) << 4) + (i & 3)] = array_in[i]
        return array
    def print_one_layer_variables(self, L, name = ""):
        for i in L:
            print(name+str(i), end = " ")
        print("\n")
    def key_schedule(self):
        MK = self.gen_variables(2)
        keyWords = [MK[0][16*i:16*(i+1)] for i in range(4)]
        keyWords += [MK[1][16*i:16*(i+1)] for i in range(4)]
        RK = []
        for r in range(self._rounds):
            rk = keyWords[0][:] + keyWords[1][:]
            keyWords[0] = keyWords[0][12:] + keyWords[0][:12]
            keyWords[1] = keyWords[1][2:] + keyWords[1][:2]
            #rotatate the words by 2
            keyWords = keyWords[2:] + keyWords[:2]
            RK.append(rk)
        return RK
    def print_variables(self):
        for r in range(self._rounds):
            print(r, ":", end = " ")
            self.print_one_layer_variables(self._variables_X[r])
            print("-----------------------S---------------------------")
            self.print_one_layer_variables(self._variables_Y[r])
            print("-----------------------L--------------------------")
            self.print_one_layer_variables(self._variables_Y[r])
            print("-----------------------add K ---------------------")
        self.print_one_layer_variables(self._variables_X[self._rounds])
    # def add_key(self, Y, X, K):
    #     return = CNF()
    # def gen_cnf(self):
    #     if self._rounds == 1:
    #         for s in range(self._number_of_sbox):
    #             variables = []
				# for i in range(self._sbox_size):
    #                 variables.append( self._variables_X[0][self._sbox_size*s)+i]
    #             for i in range(self._sbox_size):
    #                 variables.append( self._variables_Y[0][self._sbox_size*s)+i]
    #             self._complete_cnf += sbox_model(self._sbox_cnf, variables)
		# else:
			# for s in range(self._number_of_sbox):
    #                 variables = []
    #                 for i in range(self._sbox_size):
    #                     variables.append(self._variables_X[0][self._sbox_size*s)+i]
    #                 for i in range(self._sbox_size):
    #                     variables.append(self._variables_Y[0][self._sbox_size*s)+i]
    #                 self._complete_cnf += sbox_model(self._sbox_cnf, variables)
			# for r in range(1, self._rounds):
    #             if(r != self._rounds):
    #                 yy = self.P(self._variables_Y[r-1])
    #             else:
    #                 yy = self._variables_Y[r-1][:]
    #             self._complete_cnf += self.add_key(yy, self._variables_X[r], self._variables_K[r])
				# for s in range(self._number_of_sbox):
    #                 variables = []
    #                 for i in range(self._sbox_size):
    #                     variables.append(X[r][self._sbox_size*s)+i]
    #                 for i in range(self._sbox_size):
    #                     variables.append(Y[r][self._sbox_size*s)+i]
    #                 self._complete_cnf += sbox_model(self._sbox_cnf, variables)
if __name__ == "__main__":
    sbox_list = [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
    gift = CIPHER_MODEL(sbox_list, 64, 4, 2, 16, [])