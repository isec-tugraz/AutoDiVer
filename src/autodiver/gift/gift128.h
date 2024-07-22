#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
//Sbox
/* const uint8_t sbox[16] = {1,10, 4,12, 6,15, 3, 9, 2,13,11, 7, 5, 0, 8,14}; */
/* const uint8_t invsbox[16] = {13, 0, 8, 6, 2,12, 4,11,14, 7, 1,10, 3, 9,15, 5}; */
//qof
/* const uint8_t sbox[16] = {5, 14, 9, 1, 8, 10, 12, 11, 7, 6, 13, 4, 2, 0, 15, 3}; */
/* const uint8_t invsbox[16] = {13, 3, 12, 15, 11, 0, 9, 8, 4, 2, 5, 7, 6, 10, 1, 14}; */
//DEFAULT_LS
/* const uint8_t sbox[16] = {0, 3, 7, 14, 13, 4, 10, 9, 12, 15, 1, 8, 11, 2, 6, 5}; */
/* const uint8_t invsbox[16] = {0, 10, 13, 1, 5, 15, 14, 2, 11, 7, 6, 12, 8, 4, 3, 9}; */
//DEFAULT_NON_LS
/* const uint8_t sbox[16] = {1, 9, 6, 15, 7, 12, 8, 2, 10, 14, 13, 0, 4, 3, 11, 5}; */
/* const uint8_t invsbox[16] = {11, 0, 7, 13, 12, 15, 2, 4, 6, 1, 8, 14, 5, 10, 9, 3}; */
//SKINY
const uint8_t sbox[16] = {12, 6, 9, 0, 1, 10, 2, 11, 3, 8, 5, 13, 4, 14, 7, 15};
const uint8_t invsbox[16] = {3, 4, 6, 8, 12, 10, 1, 14, 9, 2, 5, 7, 0, 11, 13, 15};

//bit permutation
const uint8_t PermBits[128] = {0, 33, 66, 99, 96, 1, 34, 67, 64, 97, 2, 35, 32, 65, 98, 3, 4, 37, 70, 103, 100, 5, 38, 71, 68, 101, 6, 39, 36, 69, 102, 7, 8, 41, 74, 107, 104, 9, 42, 75, 72, 105, 10, 43, 40, 73, 106, 11, 12, 45, 78, 111, 108, 13, 46, 79, 76, 109, 14, 47, 44, 77, 110, 15, 16, 49, 82, 115, 112, 17, 50, 83, 80, 113, 18, 51, 48, 81, 114, 19, 20, 53, 86, 119, 116, 21, 54, 87, 84, 117, 22, 55, 52, 85, 118, 23, 24, 57, 90, 123, 120, 25, 58, 91, 88, 121, 26, 59, 56, 89, 122, 27, 28, 61, 94, 127, 124, 29, 62, 95, 92, 125, 30, 63, 60, 93, 126, 31};
const uint8_t PermBitsInv[128] = {0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63, 64, 69, 74, 79, 80, 85, 90, 95, 96, 101, 106, 111, 112, 117, 122, 127, 12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59, 76, 65, 70, 75, 92, 81, 86, 91, 108, 97, 102, 107, 124, 113, 118, 123, 8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55, 72, 77, 66, 71, 88, 93, 82, 87, 104, 109, 98, 103, 120, 125, 114, 119, 4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51, 68, 73, 78, 67, 84, 89, 94, 83, 100, 105, 110, 99, 116, 121, 126, 115};
// round constants
const unsigned char rc[62] = {
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
    0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
    0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
    0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
    0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13,
    0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28,
    0x10, 0x20
};

void Display_state_nibble(uint8_t *state){

  for(int i = 0; i < 32; i++){
    printf("%x", state[32 - i - 1]);
  }
  printf("\n");

}

void Display_state_bit(uint8_t *state){

  printf(" ");
  for(int i = 0; i < 32; i++){
    for(int j = 0; j < 4; j++){
      printf("%x", ((state[32 - i - 1] >> (3-j)) & 0x1));
    }
    printf(" ");
  }
  printf("\n");
}
//X0 <- x00, X1 <- x01,...., X4 <- x10, X5 <- x11,... etc.
//127,126,125,124,          .....        7,6,5,4,  3,2,1,0
void to_bits(uint8_t *A, uint8_t *B){
    for(int i = 0; i < 32; i++){
        for(int j = 0; j < 4; j++){
        B[(i * 4) + j] = (A[i] >> j) & 0x1;
        }
    }
}
void from_bits(uint8_t *A, uint8_t *B){
    //convert bit-wise variables into nibble-wise variables
    for(int i = 0; i < 32; i++){
        //0 is LSB and 3 is MSB in each nibble
        B[i]  = (A[(4 * i)]         );
        B[i] ^= (A[(4 * i) + 1] << 1);
        B[i] ^= (A[(4 * i) + 2] << 2);
        B[i] ^= (A[(4 * i) + 3] << 3);
    }
}
void SBox(uint8_t *state){
    //SBox
    for(int i=0; i<32; i++){
    	state[i] = sbox[state[i]];
    }
}
void invSBox(uint8_t *state){
    //invSBox
    for(int i=0; i<32; i++){
    	state[i] = invsbox[state[i]];
    }
}
void PLayer(uint8_t *state){
    uint8_t tmp[128];
    uint8_t bits[128];
    to_bits(state, tmp);
    for(int i = 0; i < 128; i++){
        bits[PermBits[i]] = tmp[i];
    }
    from_bits(bits, state);
}
void invPLayer(uint8_t *state){
    uint8_t tmp[128];
    uint8_t bits[128];
    to_bits(state, tmp);
    for(int i = 0; i < 128; i++){
        bits[PermBitsInv[i]] = tmp[i];
    }
    from_bits(bits, state);
}
void Key_update(uint8_t *key){
    uint8_t temp_key[32];
    //entire key>>32
    for(int i=0; i<32; i++){
        temp_key[i] = key[(i+8)%32];
    }
    for(int i=0; i<24; i++) key[i] = temp_key[i];
    //k0>>12
    key[24] = temp_key[27];
    key[25] = temp_key[24];
    key[26] = temp_key[25];
    key[27] = temp_key[26];
    //k1>>2
    key[28] = ((temp_key[28]&0xc)>>2) ^ ((temp_key[29]&0x3)<<2);
    key[29] = ((temp_key[29]&0xc)>>2) ^ ((temp_key[30]&0x3)<<2);
    key[30] = ((temp_key[30]&0xc)>>2) ^ ((temp_key[31]&0x3)<<2);
    key[31] = ((temp_key[31]&0xc)>>2) ^ ((temp_key[28]&0x3)<<2);
}
void addRk(uint8_t *state, uint8_t *key){
    uint8_t bits[128];
    uint8_t key_bits[128];
    to_bits(key, key_bits);
    to_bits(state, bits);
    int kbc=0;  //key_bit_counter
    for (int i=0; i<32; i++){
        bits[4*i+1] ^= key_bits[kbc];
        bits[4*i+2] ^= key_bits[kbc+64];
        kbc++;
    }
    from_bits(bits, state);
}
void addRc(uint8_t *state, int r){
    uint8_t bits[128];
    to_bits(state, bits);
    //add constant
    bits[3] ^= rc[r] & 0x1;
    bits[7] ^= (rc[r]>>1) & 0x1;
    bits[11] ^= (rc[r]>>2) & 0x1;
    bits[15] ^= (rc[r]>>3) & 0x1;
    bits[19] ^= (rc[r]>>4) & 0x1;
    bits[23] ^= (rc[r]>>5) & 0x1;
    bits[127] ^= 1;
    from_bits(bits, state);
}
void enc(int rounds, uint8_t *state, uint8_t *key){
    uint8_t key_copy[32];
    memcpy(key_copy, key, 32);
    /* Display_state_nibble(state); */
    //round function
    for(int r=0; r<rounds; r++){
        SBox(state);
        PLayer(state);
        addRc(state, r);
        addRk(state, key);
        Key_update(key);
    }
    memcpy(key, key_copy, 32);
}
void dec(int rounds, uint8_t *state, uint8_t *key){
    uint8_t rks[rounds][32];
    //Generate Round Keys
    for(int i=0; i<rounds; i++){
        memcpy(rks[i], key, 32);
        Key_update(key);
    }
    /* Display_state_nibble(state); */
    //round function
    for(int r=0; r<rounds; r++){
        addRc(state, rounds - 1 - r);
        addRk(state, rks[rounds - 1 - r]);
        invPLayer(state);
        invSBox(state);
        /* Display_state_nibble(state); */
    }
}
void reverse(uint8_t *A){
    uint8_t B[32];
    for(int i=0; i< 32; i++){
        B[i] = A[31-i];
    }
    for(int i=0; i< 32; i++){
        A[i] = B[i];
    }
}
int dual_enc_bit(int rounds, uint8_t *state1, uint8_t *state2, uint8_t *key, int fr, uint8_t **rks){
    int f_bit_index;
    uint8_t key_copy[32];
    uint8_t diff[32];
    memcpy(key_copy, key, 32);
    //whitening key
    addRk(state1, key);
    addRk(state2, key);
    for(int r=0; r<rounds; r++){
        SBox(state1);
        PLayer(state1);
        addRc(state1, r);
        addRk(state1, key);

        SBox(state2);
        PLayer(state2);
        addRc(state2, r);

        uint8_t temp[32];
        memcpy(temp, state2, 32);
        addRk(state2, key);
        xor_sum_32(temp, state2, rks[r]);

        Key_update(key);
        if(r == fr){
            f_bit_index = rand64() % 128;
            int f_nibble_index = f_bit_index/4;
            int index = f_bit_index % 4;
            uint8_t fault_value = (0x01 << index);
            state2[f_nibble_index] = state2[f_nibble_index] ^ fault_value;
            printf("%d %d %016LX \n",f_bit_index, f_nibble_index, fault_value);
        }
        xor_sum_32(state1, state2, diff);
        printreg(diff, 32);
    }
    /* cout << "--------------------------------------------------------------\n"; */
    memcpy(key, key_copy, 32);
    return f_bit_index;
}
int dual_enc_nibble(int rounds, uint8_t *state1, uint8_t *state2, uint8_t *key, int fr, uint8_t **rks){
    int f_nibble_index;
    uint8_t key_copy[32];
    uint8_t diff[32];
    memcpy(key_copy, key, 32);
    //whitening key
    addRk(state1, key);
    addRk(state2, key);
    for(int r=0; r<rounds; r++){
        SBox(state1);
        PLayer(state1);
        addRc(state1, r);
        addRk(state1, key);

        SBox(state2);
        PLayer(state2);
        addRc(state2, r);

        uint8_t temp[32];
        memcpy(temp, state2, 32);
        addRk(state2, key);
        xor_sum_32(temp, state2, rks[r]);

        Key_update(key);
        if(r == fr){
            f_nibble_index = rand64() % 32;
            uint64_t fault_value = rand64() & 0xF;
            while(fault_value == 0){
                fault_value = rand64() & 0xF;
            }
            state2[f_nibble_index] = state2[f_nibble_index] ^ fault_value;
            printf("%d %016LX \n", f_nibble_index, fault_value);
        }
        xor_sum_32(state1, state2, diff);
        printreg(diff, 32);
    }
    /* cout << "--------------------------------------------------------------\n"; */
    memcpy(key, key_copy, 32);
    return f_nibble_index;
}
int dual_enc_byte(int rounds, uint8_t *state1, uint8_t *state2, uint8_t *key, int fr, uint8_t **rks){
    int f_bit_index;
    int offset;
    uint64_t fault_value;
    uint8_t fault_state[32];
    uint8_t key_copy[32];
    uint8_t diff[32];
    memcpy(key_copy, key, 32);
    fr = fr - 1;
    for(int r=0; r<rounds; r++){
        SBox(state1);
        PLayer(state1);
        addRc(state1, r);
        addRk(state1, key);

        SBox(state2);
        PLayer(state2);
        addRc(state2, r);

        uint8_t temp[32];
        memcpy(temp, state2, 32);
        addRk(state2, key);
        xor_sum_32(temp, state2, rks[r]);

        Key_update(key);

        if(r == fr){
            f_bit_index = rand64() % 56;
            offset = rand64() % 2;

            fault_value = rand64() & 0x00000000000000FFUL;
            while(fault_value == 0){
                fault_value = rand64() & 0x00000000000000FFUL;
            }

            printf("%d %d %016LX \n", f_bit_index, offset, fault_value);
            fault_value = fault_value << f_bit_index;

            memset(fault_state, 0x00, 32);
            if(offset > 0){
                offset = 16;
            }
            for(int i=0; i<16; i++){
                fault_state[offset + i] = (fault_value >> (4*i)) & 0x0F;
            }
            xor_sum_32(state2, fault_state, state2);
            printf("%d %d %016LX \n", f_bit_index, offset, fault_value);
            printreg(fault_state, 32);
        }
        xor_sum_32(state1, state2, diff);
        printreg(diff, 32);
    }
    cout << "--------------------------------------------------------------\n";
    memcpy(key, key_copy, 32);
    return f_bit_index;
}
uint16_t circ_left_shift(uint16_t x,int pos){
    return ((((x<<pos)&0xffff)|((x>>(16-pos))&0xffff))&0xffff);
    }

void new_inv_key_schedule(uint8_t *k39, uint8_t *k38, uint8_t *k37, uint8_t *k36){
    uint32_t u1 = 0UL,
             v1 = 0UL,
             u2 = 0UL,
             v2 = 0UL;

    /* constructing u,v of 32 bits */
    for(int8_t i=0;i<32;i++){
        u1 = ((u1<<1)|((k39[31-i] >> 2)&0x1));
        v1 = ((v1<<1)|((k39[31-i] >> 1)&0x1));

        u2 = ((u2<<1)|((k38[31-i] >> 2)&0x1));
        v2 = ((v2<<1)|((k38[31-i] >> 1)&0x1));
        }

    /* forming k[] w.r.to second last round */
    uint16_t k[8];
    k[0] = (v2&0xffff);
    k[1] = ((v2>>16)&0xffff);
    k[4] = (u2&0xffff);
    k[5] = ((u2>>16)&0xffff);

    k[2] = (v1&0xffff);
    k[3] = ((v1>>16)&0xffff);
    k[6] = (u1&0xffff);
    k[7] = ((u1>>16)&0xffff);


    uint16_t prev_k[8] = {0};
    prev_k[0] = circ_left_shift(k[6],12);
    prev_k[1] = circ_left_shift(k[7],2);
    prev_k[4] = k[2];
    prev_k[5] = k[3];

    prev_k[2] = circ_left_shift(k[4],12);
    prev_k[3] = circ_left_shift(k[5],2);
    prev_k[6] = k[0];
    prev_k[7] = k[1];
/* finding prev round key */    uint32_t prev_u,
             prev_v;
    /* to retrieve the third last round key */    prev_v = (prev_k[1]<<16)|prev_k[0];
    prev_u = (prev_k[5]<<16)|prev_k[4];
    /* storing the third last round key */    for(int i=0;i<32;i++){
        k37[i] = (((prev_u>>i)&1) << 2)|(((prev_v>>i)&1) << 1);        }

    /* to retrieve the fourth last round key */    prev_v = (prev_k[3]<<16)|prev_k[2];
    prev_u = (prev_k[7]<<16)|prev_k[6];
    /* storing the fourth last round key */    for(int i=0;i<32;i++){
        k36[i] = (((prev_u>>i)&1) << 2)|(((prev_v>>i)&1) << 1);        }
}
void inv_master_key_schedule(uint8_t *k39, uint8_t *k38, uint8_t *k0){
    for (int8_t i=0; i<19; i++){
        uint32_t u1 = 0UL,
                 v1 = 0UL,
                 u2 = 0UL,
                 v2 = 0UL;

        /* constructing u,v of 32 bits */
        for(int8_t i=0;i<32;i++){
            u1 = ((u1<<1)|((k39[31-i] >> 2)&0x1));
            v1 = ((v1<<1)|((k39[31-i] >> 1)&0x1));

            u2 = ((u2<<1)|((k38[31-i] >> 2)&0x1));
            v2 = ((v2<<1)|((k38[31-i] >> 1)&0x1));
            }

        /* forming k[] w.r.to second last round */
        uint16_t k[8];
        k[0] = (v2&0xffff);
        k[1] = ((v2>>16)&0xffff);
        k[4] = (u2&0xffff);
        k[5] = ((u2>>16)&0xffff);

        k[2] = (v1&0xffff);
        k[3] = ((v1>>16)&0xffff);
        k[6] = (u1&0xffff);
        k[7] = ((u1>>16)&0xffff);


        uint16_t prev_k[8] = {0};
        prev_k[0] = circ_left_shift(k[6],12);
        prev_k[1] = circ_left_shift(k[7],2);
        prev_k[4] = k[2];
        prev_k[5] = k[3];

        prev_k[2] = circ_left_shift(k[4],12);
        prev_k[3] = circ_left_shift(k[5],2);
        prev_k[6] = k[0];
        prev_k[7] = k[1];
/* finding prev round key */        uint32_t prev_u,
                 prev_v;
        /* to retrieve the third last round key */        prev_v = (prev_k[1]<<16)|prev_k[0];
        prev_u = (prev_k[5]<<16)|prev_k[4];
        /* storing the third last round key */        for(int i=0;i<32;i++){
            k39[i] = (((prev_u>>i)&1) << 2)|(((prev_v>>i)&1) << 1);            }
        /* to retrieve the fourth last round key */
        prev_v = (prev_k[3]<<16)|prev_k[2];        prev_u = (prev_k[7]<<16)|prev_k[6];
        /* storing the fourth last round key */
        for(int i=0;i<32;i++){            k38[i] = (((prev_u>>i)&1) << 2)|(((prev_v>>i)&1) << 1);
            }
        /* if the round key for 0th round and 1st round has derived */        if (i == 18){
            /* rearranging the master key words from perv key */            uint16_t new_k[8] = {0};
            new_k[0] = prev_k[2];   new_k[1] = prev_k[3];            new_k[2] = prev_k[0];   new_k[3] = prev_k[1];
            new_k[4] = prev_k[6];   new_k[5] = prev_k[7];            new_k[6] = prev_k[4];   new_k[7] = prev_k[5];
            /* storing the master key from 16-bit new key array */
            for (int j=0; j<8; j++){                for (int p=0; p<4; p++){
                    k0[4*j +p] = (new_k[j] >> (4*p))&0xf;                }}}}
}
