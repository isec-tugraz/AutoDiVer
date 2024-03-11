from ..util import IndexSet
class GiftIndices(IndexSet):
    def __init__(self, numrounds):
        super().__init__()
        self.numrounds = numrounds
        # subbytes, permute bits, add round_key
        # index: round, sbox, bit
        self.add_index_array('sbox_in_val', (self.numrounds, 16, 4))
        self.add_index_array('sbox_out_val', (self.numrounds, 16, 4))