def Sb0(bitStringArray):
    x = int("".join((map(str, bitStringArray))), 2)
    if x == 0:
        result = int("c", 16)
    elif x == 1:
        result = int("a", 16)
    elif x == 2:
        result = int("d", 16)
    elif x == 3:
        result = 3
    elif x == 4:
        result = int("e", 16)
    elif x == 5:
        result = int("d", 16)
    elif x == 6:
        result = int("f", 16)
    elif x == 7:
        result = 7
    elif x == 8:
        result = 8
    elif x == 9:
        result = 9
    elif x == int("a", 16):
        result = 1
    elif x == int("b", 16):
        result = 5
    elif x == int("c", 16):
        result = 0
    elif x == int("d", 16):
        result = 2
    elif x == int("e", 16):
        result = 4
    elif x == int("f", 16):
        result = 6
    else:
        raise Exception("The value {} does not have a corresponding value in the lookup table".format(x))
    b0 = (result >> 3) & 1
    b1 = (result >> 2) & 1
    b2 = (result >> 1) & 1
    b3 = result & 1
    newBitStringArray = list(map(str, [b0, b1, b2, b3]))
    return newBitStringArray


def Sb1(bitStringArray):
    x = int("".join((map(str, bitStringArray))), 2)
    # print(bitStringArray, "=>", x)
    if x == 0:
        result = 1
    elif x == 1:
        result = 0
    elif x == 2:
        result = 5
    elif x == 3:
        result = 3
    elif x == 4:
        result = int("e", 16)
    elif x == 5:
        result = 2
    elif x == 6:
        result = int("f", 16)
    elif x == 7:
        result = 7
    elif x == 8:
        result = int("d", 16)
    elif x == 9:
        result = int("a", 16)
    elif x == int("a", 16):
        result = 9
    elif x == int("b", 16):
        result = int("b", 16)
    elif x == int("c", 16):
        result = int("c", 16)
    elif x == int("d", 16):
        result = 8
    elif x == int("e", 16):
        result = 4
    elif x == int("f", 16):
        result = 6
    else:
        raise Exception("The value {} does not have a corresponding value in the lookup table".format(x))
    b0 = (result >> 3) & 1
    b1 = (result >> 2) & 1
    b2 = (result >> 1) & 1
    b3 = result & 1
    newBitStringArray = list(map(str, [b0, b1, b2, b3]))
    return newBitStringArray


def SSbi(x, i):
    """
    Implements the Sbox within midori.
    :param x: The integer that is fed into the sbox
    :param i: Indicates which sbox must be used
    :return: A permuted x
    """
    # x0 is the most significant bit, x7 the least significant
    x0 = (x >> 7) & 1
    x1 = (x >> 6) & 1
    x2 = (x >> 5) & 1
    x3 = (x >> 4) & 1
    x4 = (x >> 3) & 1
    x5 = (x >> 2) & 1
    x6 = (x >> 1) & 1
    x7 = x & 1
    binResult = []
    if i == 0:
        A = [x4, x1, x6, x3]
        B = [x0, x5, x2, x7]
        # print(A + B)
        n0 = Sb1(A)
        n1 = Sb1(B)
        binResult = [n1[0], n0[1], n1[2], n0[3],
                     n0[0], n1[1], n0[2], n1[3]]
        # n0 = Sb1([x0, x1, x2, x3])
        # n1 = Sb1([x4, x5, x6, x7])
        # binResult = [n0[0], n0[1], n0[2], n0[3],
        #              n1[0], n1[1], n1[2], n1[3]]
    elif i == 1:
        n0 = Sb1([x1, x6, x7, x0])
        n1 = Sb1([x5, x2, x3, x4])
        binResult = [n0[3], n0[0], n1[1], n1[2],
                     n1[3], n1[0], n0[1], n0[2]]
    elif i == 2:
        n0 = Sb1([x2, x3, x4, x1])
        n1 = Sb1([x6, x7, x0, x5])
        binResult = [n1[2], n0[3], n0[0], n0[1],
                     n0[2], n1[3], n1[0], n1[1]]
    elif i == 3:
        n0 = Sb1([x7, x4, x1, x2])
        n1 = Sb1([x3, x0, x5, x6])
        binResult = [n1[1], n0[2], n0[3], n1[0],
                     n0[1], n1[2], n1[3], n0[0]]
    result = int("".join(binResult), 2)
    return result


def sbox128():
    SSb0 = [hex(SSbi(i, 0)) for i in range(16)]
    print(SSb0)

if __name__ == "__main__":
    sbox128()
