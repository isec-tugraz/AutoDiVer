#include <stdio.h>
#include<stdlib.h>
#include<stdint.h>
#define STATE_SIZE 64
#define KEY_SIZE 128
#define NO_OF_ROUNDS 25

/* ---------------------------------------------------------------------------------------- */
/* gist: oracle for rectangle-80 */
/* ---------------------------------------------------------------------------------------- */

/* If the print details is true then only all the state values will print */
char *print_details = "false";
/* char* print_details = "true"; */

/* #define CYTHON */

uint64_t sbox(uint64_t msg){
    /* sbox table of rectangle cipher */
    uint8_t sbox_table[16] = {0x6,0x5,0xC,0xA,0x1,0xE,0x7,0x9,0xB,0x0,0x3,0xD,0x8,0xF,0x4,0x2};

    /* extracting the state into 4 rows i.e. msg = row[3]||row[2]||row[1]||row[0] */
    uint16_t row[4];
    for (uint8_t i=0; i<4; i++){
        row[i] = (msg >> (16*i)) & 0xffff;
    }
    /* extracting first four cols to apply the sbox */
    uint8_t col[16] = {0};
    for (int8_t j=15; j>=0; j--){
        col[j] = (((row[3]>>j)&1)<<3) | (((row[2]>>j)&1)<<2) | (((row[1]>>j)&1)<<1) | (((row[0]>>j)&1)<<0);
    }

    /* applying sbox on each col nibble */
    for(int8_t j=15;j>=0;j--){
        col[j] = sbox_table[col[j]];;
    }

    /* making all row[]'s to 0 */
    for (uint8_t i=0; i<4; i++){
        row[i] = 0;
    }
    /* making rows from the updated cols */
    for(int8_t j=15;j>=0;j--){
        row[3] |= ((col[j]>>3)&1)<<j;
        row[2] |= ((col[j]>>2)&1)<<j;
        row[1] |= ((col[j]>>1)&1)<<j;
        row[0] |= ((col[j]>>0)&1)<<j;
    }

    msg = 0UL;
    for (int8_t i=3; i>=0; i--){
        msg = (msg<<16) | (row[i]&0xffff);
    }

    /* for printing purpose */
    if (print_details == "true"){
        printf("after sbox:\t");
        printf("%016lX \n", msg);
    }
    return msg;
}


/* left shift within 16-bit only */
uint32_t circ_left_shift32(uint32_t x, int pos){
    return ((x<<pos)|(x>>(32-pos)));
}
uint16_t circ_left_shift16(uint16_t x, int pos){
    return ((x<<pos)|(x>>(16-pos)));
}
uint64_t shift_row(uint64_t msg){
    uint16_t row[4];
    for (uint8_t i=0; i<4; i++){
        row[i] = (msg >> (16*i)) & 0xffff;
    }

    row[1] = circ_left_shift16(row[1], 1);
    row[2] = circ_left_shift16(row[2], 12);
    row[3] = circ_left_shift16(row[3], 13);

    msg = 0UL;
    for (int8_t i=3; i>=0; i--){
        msg = (msg<<16) | (row[i]&0xffff);
    }

    /* for printing purpose */
    if (print_details == "true"){
        printf("after sr:\t");
        printf("%016LX \n", msg);
    }
    return msg;
}


uint64_t extract_round_key(uint64_t *key){
    uint64_t round_key = 0UL;
    round_key = (round_key << 16) | ((key[1] >> 32) & 0xFFFF);
    round_key = (round_key << 16) |  (key[1]        & 0xFFFF);
    round_key = (round_key << 16) | ((key[0] >> 32) & 0xFFFF);
    round_key = (round_key << 16) |  (key[0]        & 0xFFFF);

    return round_key;
}
void ksp(uint64_t *round_key, uint64_t *key){
    /* sbox table of rectangle */
    uint8_t sbox_table[16] = {0x6,0x5,0xC,0xA,0x1,0xE,0x7,0x9,0xB,0x0,0x3,0xD,0x8,0xF,0x4,0x2};
    /* --------------------------------------------------------------------------- */
    /* for adding the round constants */
    /* --------------------------------------------------------------------------- */
    uint8_t rc[25] = {0x01,0x02,0x04,0x09,0x12,0x05,0x0b,0x16,0x0c,0x19,0x13,0x07,0x0f,0x1f,
                      0x1e, 0x1c, 0x18, 0x11, 0x03, 0x06, 0x0d, 0x1b, 0x17, 0x0e, 0x1d};

    for (uint8_t round=0; round<NO_OF_ROUNDS; round++){
        round_key[round] = extract_round_key(key);

        uint32_t row[4];
        row[0] =   key[0]      & 0xffffffff;
        row[1] =  (key[0]>>32) & 0xffffffff;
        row[2] =   key[1]      & 0xffffffff;
        row[3] =  (key[1]>>32) & 0xffffffff;

        /* extracting first four cols to apply the sbox */
        uint8_t col[8] = {0};
        for (int8_t j=7; j>=0; j--){
            col[j]=(((row[3]>>j)&1)<<3)|(((row[2]>>j)&1)<<2)|(((row[1]>>j)&1)<<1)|(((row[0]>>j)&1)<<0);
        }

        /* applying sbox to the right 8 columns */
        for (int8_t j=0; j<8; j++){
            col[j] = sbox_table[col[j]];
        }

        for (int8_t i=3; i>=0; i--){
            /* ommiting the first 8 col vals for updation */
            row[i] &= 0xffffff00;
            /* updating the rows depending upon the col vals */
            for(int j=7; j>=0; j--){
                row[i] |= ((col[j] >> i) & 0x01) << j;
            }
        }

        /* updation of the rows */
        uint32_t new_row[4];
        new_row[0] = circ_left_shift32(row[0], 8)^row[1];
        new_row[1] = row[2];
        new_row[2] = circ_left_shift32(row[2], 16)^row[3];
        new_row[3] = row[0];


        /* xoring the rc val */
        new_row[0] ^= rc[round]&0x1f;

        /* making the key val to 0 for updating the key */
        key[0] = (uint64_t)(new_row[0]&0xffffffff);
        key[0] |= (uint64_t)(new_row[1]&0xffffffff) << 32;
        key[1] = (uint64_t)(new_row[2]&0xffffffff);
        key[1] |= (uint64_t)(new_row[3]&0xffffffff) << 32;
    }

    /* extracting the round key from the updated key */
    round_key[NO_OF_ROUNDS] = extract_round_key(key);

    /* for printing purpose */
    if (print_details == "true"){
        printf("********************************************************************************\n");
        printf("round keys:\n");
        printf("********************************************************************************\n");

        for(uint8_t round=0; round<=NO_OF_ROUNDS; round++){
            printf("for round %d:\t", round);
            printf("%016lx \n", round_key[round]);
        }
    }
}


uint64_t add_round_key(uint64_t msg, uint64_t rkey){
    return (msg ^ rkey);
}


uint64_t rectangle_encrypt(uint64_t msg, uint64_t *orkey, int rounds){
    /* allocating mem for round keys */
    uint64_t *round_key = (uint64_t *)malloc((NO_OF_ROUNDS+1)*sizeof(uint64_t));
    uint64_t key[2];
    key[0] = orkey[0];
    key[1] = orkey[1];
    /* generating the round keys */
    ksp(round_key, key);

    /* round function */
    for(uint8_t round=0; round < rounds; round++){
        /* for printing purpose */
        if (print_details == "true"){
            printf("\n");
            printf("****************************************************************************\n");
            printf("for round %d:\n", round);
            printf("****************************************************************************\n");
            printf("%016LX \n", msg);
        }

        msg = add_round_key(msg, round_key[round]);
        msg = sbox(msg);
        msg = shift_row(msg);
    }
    msg = add_round_key(msg, round_key[rounds]);
    free(round_key);
    return msg;
}
uint64_t rectangle_encrypt_long_key(uint64_t msg, uint64_t *round_key, int rounds){
    for(uint8_t round=0; round < rounds; round++){
        msg = add_round_key(msg, round_key[round]);
        msg = sbox(msg);
        msg = shift_row(msg);
    }
    msg = add_round_key(msg, round_key[rounds]);
    return msg;
}
void nibbleToKey(uint8_t *arr, uint64_t *key){
    uint64_t S[4] = {0UL, 0UL, 0UL, 0UL};
    for (int i=31; i>=0; i--) {
        for (int j=3; j>=0; j--) {
            S[j] = (S[j] << 1) | ((arr[i] >> j) & 0x1);
        }
    }
    key[0] = S[0] | (S[1] << 32);
    key[1] = S[2] | (S[3] << 32);
}
uint64_t nibbleToState(uint8_t *arr){
    uint64_t state = 0UL;
    uint64_t S[4] = {0UL, 0UL, 0UL, 0UL};
    for (int i=15; i>=0; i--) {
        for (int j=3; j>=0; j--) {
            S[j] = (S[j] << 1) | ((arr[i] >> j) & 0x1);
        }
    }
    for (int i=3; i>=0; i--) {
        state = (state << 16) | S[i];
    }
    return state;
}

#ifndef CYTHON
void testConversion(){
    uint8_t arr[32] = {2, 14, 11,  5,  5,  3, 11,  8, 10,  7,  2,  5, 10, 12,  7,  7,  1,
        4, 12,  9,  9,  4, 13, 11,  7,  8,  5,  2, 14,  5, 14,  4};
    uint64_t key[2];
    nibbleToKey(arr, key);
    printf("IN C %016lx, %016lx \n", key[0], key[1]);
}
void testVectors0(){
    uint64_t msg = 0x0UL;
    uint64_t key[2];
    key[0] = 0x0UL;
    key[1] = 0x0UL;
    uint64_t cip = rectangle_encrypt(msg, key, NO_OF_ROUNDS);
    printf("%016LX \n", cip);
}
void testVectors(){
    uint64_t msg = 0xffffffffffffffffUL;
    uint64_t key[2];
    key[0] = 0xffffffffffffffffUL;
    key[1] = 0xffffffffffffffffUL;
    uint64_t cip = rectangle_encrypt(msg, key, NO_OF_ROUNDS);
    printf("%016LX \n", cip);
}
int main(){
    uint64_t msg = 0x0123456789ABCDEFUL;
    uint64_t key[2];
    key[0] = 0x0123456789ABCDEFUL;
    key[1] = 0x0123456789ABCDEFUL;
    uint64_t cip = rectangle_encrypt(msg, key, NO_OF_ROUNDS);
    printf("%016LX \n", cip);
    testVectors();
    testVectors0();
    testConversion();
}
#endif
