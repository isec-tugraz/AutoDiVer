import binascii, sys
import binascii as B
if sys.version_info.major == 2:
  def b2s(x):
    """convert bytes to str"""
    return x
  def s2b(x):
    """convert str to bytes"""
    return x
else:
  def b2s(x):
    """convert bytes to str"""
    if isinstance(x, str): return x
    return x.decode("utf8")
  def s2b(x):
    """convert str to bytes"""
    if isinstance(x, bytes): return x
    return x.encode("utf8")
  xrange = range
def xor_key(state, key):
  """
  XOR key
  state:    state as integer
  key:      key as integer
  returns:  new state as integer
  """
  return state ^ key
whiten        = xor_key   # key whitening
add_roundkey  = xor_key   # add round key
def p_layer(state, inv = False):
  """
  apply permutation matrix P
  state:    state as integer
  returns:  new state as integer
  """
  p = P if not inv else P_inv
  state_ = 0
  for i in xrange(64):
    state_ |= ((state >> i) & 0b1) << p[i]
  return state_
def p_layer_inv(state):
  """apply permutation matrix P_inv"""
  return p_layer(state, True)
def s_layer(state, inv = False):
  """
  apply S-box S
  state:    state as integer
  returns:  new state as integer
  """
  s = S if not inv else S_inv
  state_ = 0
  for i in xrange(16):
    state_ |= s[(state >> (i*4)) & 0xF] << (i*4)
  return state_
def s_layer_inv(state):
  """apply S-box S_inv"""
  return s_layer(state, True)
def l_layer(state, inv = False):
  """
  apply linear mappings L[0-3]
  state:    state as integer
  returns:  new state as integer
  """
  l = [L3,L2,L1,L0] if not inv else [L3_inv,L2_inv,L1_inv,L0_inv]
  state_ = 0
  for i in xrange(4):
    state_ |= matrix_mult(l[i], (state >> (i*16)) & 0xFFFF) << (i*16)
  return state_
def l_layer_inv(state):
  """apply linear mappings L[0-3]_inv"""
  return l_layer(state, True)
def f(i, k1):
  """
  round key (see specification)
  i:        round number (1 <= i <= rounds)
  k1:       round key basis as bytes
  returns:  round key as integer
  """
  return b2i(b"".join(
    i2b(g(b2i(k1[j]), i, j // 2)) if j%2 else i2b(k1[j])
      for j in xrange(8)
  ))
def g(x, i, j):
  """
  dynamic part of round key (see specification)
  x:        key part (1 byte) as integer
  i:        round number
  j:        part number (0 <= j <= 3)
  returns:  new key part (1 byte) as integer
  """
  m = { 0: 193, 1: 165, 2: 81, 3: 197 }
  return (x + m[j]*i) % 256
def p():
  """
  permutation matrix (see specification)
  returns: array of indices as integers
  """
  m = [None]*64
  for i in xrange(4):
    for j in xrange(16):
      m[63-(j*4+i)] = (3-i)*16+(15-j)
  return m
P     = p()
S     = [ 0x0, 0x4, 0x8, 0xF, 0x1, 0x5, 0xE, 0x9,
          0x2, 0x7, 0xA, 0xC, 0xB, 0xD, 0x6, 0x3 ]
P_inv = [ P.index(_i) for _i in xrange(64) ]
S_inv = [ S.index(_i) for _i in xrange(16) ]
# L[0-3]{,_inv}                                                 # {{{1
L0 = [
  0b0000100010001000, 0b0000010001000100, 0b0000001000100010,
  0b0000000100010001, 0b1000000010001000, 0b0100000001000100,
  0b0010000000100010, 0b0001000000010001, 0b1000100000001000,
  0b0100010000000100, 0b0010001000000010, 0b0001000100000001,
  0b1000100010000000, 0b0100010001000000, 0b0010001000100000,
  0b0001000100010000,
]
L0_inv = L0
L1 = [
  0b1100000000010000, 0b0110000000001000, 0b0011000000000100,
  0b0001100000000010, 0b0000110000000001, 0b0000011010000000,
  0b0000001101000000, 0b1000000100100000, 0b1000000000011000,
  0b0100000000001100, 0b0010000000000110, 0b0001000000000011,
  0b0000100010000001, 0b0000010011000000, 0b0000001001100000,
  0b0000000100110000,
]
L1_inv = [
  0b0000001100000010, 0b1000000100000001, 0b1100000010000000,
  0b0110000001000000, 0b0011000000100000, 0b0001100000010000,
  0b0000110000001000, 0b0000011000000100, 0b0001000000011000,
  0b0000100000001100, 0b0000010000000110, 0b0000001000000011,
  0b0000000110000001, 0b1000000011000000, 0b0100000001100000,
  0b0010000000110000,
]
L2 = [
  0b0000110000000001, 0b0000011010000000, 0b0000001101000000,
  0b1000000100100000, 0b1100000000010000, 0b0110000000001000,
  0b0011000000000100, 0b0001100000000010, 0b0000100010000001,
  0b0000010011000000, 0b0000001001100000, 0b0000000100110000,
  0b1000000000011000, 0b0100000000001100, 0b0010000000000110,
  0b0001000000000011,
]
L2_inv = [
  0b0011000000100000, 0b0001100000010000, 0b0000110000001000,
  0b0000011000000100, 0b0000001100000010, 0b1000000100000001,
  0b1100000010000000, 0b0110000001000000, 0b0000000110000001,
  0b1000000011000000, 0b0100000001100000, 0b0010000000110000,
  0b0001000000011000, 0b0000100000001100, 0b0000010000000110,
  0b0000001000000011,
]
L3 = [
  0b1000100000001000, 0b0100010000000100, 0b0010001000000010,
  0b0001000100000001, 0b1000100010000000, 0b0100010001000000,
  0b0010001000100000, 0b0001000100010000, 0b0000100010001000,
  0b0000010001000100, 0b0000001000100010, 0b0000000100010001,
  0b1000000010001000, 0b0100000001000100, 0b0010000000100010,
  0b0001000000010001,
]
L3_inv = L3
                                                                # }}}1
def matrix_mult(m, v):
  """binary multiplication of 16x16 matrix w/ vector"""
  w = 0
  for i,r in enumerate(m):
    w |= (bin(r & v)[2:].count('1') % 2) << (15 - i)
  return w
def b2i(x):
  """convert bytes to integer"""
  if isinstance(x, int): return x
  return int(binascii.hexlify(x), 16)
def i2b(x, n = 1):
  """convert integer to bytes of length (at least) n"""
  if isinstance(x, bytes): return x
  return binascii.unhexlify(s2b("%0*x" % (n*2,x)))
class Pride(object):                                            # {{{1
  """PRIDE cipher"""
  def __init__(self, key, rounds = 20):
    """
    Create a PRIDE cipher object
    key:      the key as a 128-bit bytes
    rounds:   the number of rounds as an integer (32 by default)
    """
    if len(key) != 16:
      raise ValueError("Key must be a 128-bit bytes")
    self.k0, k1     = b2i(key[:8]), key[8:]
    self.rounds     = rounds
    self.roundkeys  = [ f(i+1, k1) for i in xrange(rounds) ]
  def encrypt(self, block):
    """
    Encrypt 1 block (8 bytes)
    block:    plaintext block as bytes
    returns:  ciphertext block as bytes
    """
    state = b2i(block)
    state = p_layer_inv(state)
    state = whiten(state, self.k0)
    for i in xrange(self.rounds):
      state = add_roundkey(state, p_layer_inv(self.roundkeys[i]))
      state = s_layer(state)
      if i != self.rounds - 1:
        state = p_layer(state)
        state = l_layer(state)
        state = p_layer_inv(state)
    state = whiten(state, self.k0)  # k2 = k0
    state = p_layer(state)
    return i2b(state, 8)
  def decrypt(self, block):
    """
    Decrypt 1 block (8 bytes)
    block:    ciphertext block as bytes
    returns:  plaintext block as bytes
    """
    state = b2i(block)
    state = p_layer_inv(state)
    state = whiten(state, self.k0)  # k2 = k0
    for i in xrange(self.rounds):
      state = s_layer_inv(state)
      state = add_roundkey(state, p_layer_inv(self.roundkeys[-i-1]))
      if i != self.rounds - 1:
        state = p_layer(state)
        state = l_layer_inv(state)
        state = p_layer_inv(state)
    state = whiten(state, self.k0)
    state = p_layer(state)
    return i2b(state, 8)
  def get_block_size(self):
    return 8
if __name__ == "__main__":
    key1        = B.unhexlify(b"00000000000000000000000000000000")
    plain1      = B.unhexlify(b"0000000000000000")
    PP     = Pride(key1)
    encrypted1  = PP.encrypt(plain1)
    C = b2s(B.hexlify(encrypted1))  # b2s so it works w/ python 2 and 3
    print(C)
    decrypted1  = PP.decrypt(encrypted1)
    D = b2s(B.hexlify(decrypted1))
    print(D)