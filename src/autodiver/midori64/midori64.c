#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define ROUNDS 16

uint64_t sbox(uint64_t msg){
    /* sbox table of midori-64 cipher */
    uint8_t sbox_table[16] = {0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7,
                              0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6};
    uint64_t cip = 0UL;

    /* applying sbox on each nibble */
    for(int8_t i=15;i>=0;i--){
        uint8_t nibble = sbox_table[(msg >> 4*i)&0xf];;
        cip = (cip << 4)|(nibble&0xf);
    }
    return cip;
}

uint64_t sr(uint64_t msg){
    /* shift row index of midrori state */
    uint8_t sr_index[16] = {0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8};
    uint64_t cip = 0UL;

    /* for each nibble */
    for(int i=0;i<16;i++){
        uint8_t nibble = 0;
        /* taking the nibble according to sr index */
        nibble = (msg>> (60 - 4*sr_index[i]))&0xf;
        cip = (cip<<4) | (nibble&0xf);
    }
    return cip;
}


uint64_t mc(uint64_t msg){
    uint64_t cip = 0UL;

    for (uint8_t col=0; col<4; col++){
        /* storing nibbles into nibble[4].
         * The nibbles are stored as msg = n0||n1||n2||n3 */
        uint8_t nibble[4] = {0};
        for(uint8_t idx=0; idx<4; idx++){
            nibble[3-idx] = (msg>>(4*idx + 16*col))&0xf;
        }
        uint64_t r[4] = {0};
        r[0] = nibble[1]^nibble[2]^nibble[3];
        r[1] = nibble[0]^nibble[2]^nibble[3];
        r[2] = nibble[0]^nibble[1]^nibble[3];
        r[3] = nibble[0]^nibble[1]^nibble[2];

        /* storing resultant nibbles into msg */
        uint64_t tmp = (r[0]<<(12  + 16*col)) | (r[1]<<(8  + 16*col)) |
                       (r[2]<<( 4  + 16*col)) | (r[3]<<(0  + 16*col));
        cip |= tmp;
    }
    return cip;
}


uint64_t add_round_key(uint64_t msg, uint64_t key){
    return (msg^key);
}


void generate_rnd_key(const uint64_t *key, uint64_t *rnd_key, int rounds){
    uint64_t alpha[] = {0x0001010110110011, 0x0111100011000000, 0x1010010000110101,
                        0x0110001000010011, 0x0001000001001111, 0x1101000101110000,
                        0x0000001001100110, 0x0000101111001100, 0x1001010010000001,
                        0x0100000010111000, 0x0111000110010111, 0x0010001010001110,
                        0x0101000100110000, 0x1111100011001010, 0x1101111110010000,
                        0x0111110010000001};

    /* getting the round keys */
    /* for (uint8_t i = 0; i < rounds - 1; i++){ */
    for (int i = 0; i < rounds; i++){
        if (!(i < sizeof(alpha)/sizeof(alpha[0]))) {
            fprintf(stderr, "not enough round constants");
            abort();
        }

        rnd_key[i] = key[i % 2] ^ alpha[i];
        /* rnd_key[i] = key[i % 2]; */
    }
}

uint64_t enc_midori64(uint64_t msg, const uint64_t *key, int rounds){
    if (rounds > ROUNDS) {
        fprintf(stderr, "rounds should be less than or equal to %d\n", ROUNDS);
        abort();
    }

    /* allocating mem for round keys */
    uint64_t rnd_key[ROUNDS];
    generate_rnd_key(key, rnd_key, rounds);

    if (rounds == 0) {
      return msg;
    }

    uint64_t cip = msg;

    cip = cip ^ key[0] ^ key[1];
    /* printf("W %016lX \n", cip); */
    for (uint8_t i = 0; i < rounds - 1; i++){
        cip = sbox(cip);
        cip = sr(cip);
        cip = mc(cip);
        /* printf("%d %016lX \n", i, cip); */
        /* printf("K %016lX \n", rnd_key[i]); */
        cip = add_round_key(cip, rnd_key[i]);
        /* printf("%d %016lX \n", i, cip); */
    }

    cip = sbox(cip);
    cip = cip ^ key[0] ^ key[1];

    return cip;
}

// int main(){
//     uint64_t msg;
//     uint64_t cip;
//     uint64_t key[2];
//
//     msg = 0x42c20fd3b586879e;
//     key[0] = 0x687ded3b3c85b3f3;
//     key[1] = 0x5b1009863e2a8cbf;
//
//     /* msg[0] = 0x0; */
//     /* insert(key, 0x0, 0x0); */
//
//     cip = enc_midori64(msg, key);
//     printf("%016lX \n", msg);
//     printf("%016lX",key[0]);
//     printf("%016lX \n",key[1]);
//     printf("%016lX \n", cip);
// }
