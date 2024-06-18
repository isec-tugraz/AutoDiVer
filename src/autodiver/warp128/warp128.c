/*********************************************
 * Reference implementation by WARP Team     *
**********************************************/
#include <stdio.h>
#include <stdint.h>
#define R 41    /*round number*/
#define RN    6    /*rotation number*/
#define BR    32  /*brunch number*/
#define BR_HALF    (BR / 2)  /*half of the branch number*/
#define PRINT_INTER 0
int Sbox[BR_HALF] = { 0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6 };
int perm[BR] = { 31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10, 15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26, };
int RC0[R] = { 0x0U, 0x0U, 0x1U, 0x3U, 0x7U, 0xfU, 0xfU, 0xfU, 0xeU, 0xdU, 0xaU, 0x5U, 0xaU, 0x5U, 0xbU, 0x6U, 0xcU, 0x9U, 0x3U, 0x6U, 0xdU, 0xbU, 0x7U, 0xeU, 0xdU, 0xbU, 0x6U, 0xdU, 0xaU, 0x4U, 0x9U, 0x2U, 0x4U, 0x9U, 0x3U, 0x7U, 0xeU, 0xcU, 0x8U, 0x1U, 0x2U};
int RC1[R] = { 0x4U, 0xcU, 0xcU, 0xcU, 0xcU, 0xcU, 0x8U, 0x4U, 0x8U, 0x4U, 0x8U, 0x4U, 0xcU, 0x8U, 0x0U, 0x4U, 0xcU, 0x8U, 0x4U, 0xcU, 0xcU, 0x8U, 0x4U, 0xcU, 0x8U, 0x4U, 0x8U, 0x0U, 0x4U, 0x8U, 0x0U, 0x4U, 0xcU, 0xcU, 0x8U, 0x0U, 0x0U, 0x4U, 0x8U, 0x4U, 0xcU};
void print_state(uint8_t *m){
    for (int i = 0; i < BR; i++){
        printf("%x, ", m[i]&0xf);
    }
    printf("\n");
};
void printState(uint8_t *state){
    printf("L: ");
    for (int x = 0; x < BR_HALF; x++){
        printf("%x ", state[2 * x + 0]);
    }
    printf("R: ");
    for (int x = 0; x < BR_HALF; x++){
        printf("%x ", state[2 * x + 1]);
    }
    printf("\n");
}
void sboxkey(uint8_t *state, uint8_t *k, int r){
    for (int i = 0; i < BR_HALF; i++){
        state[i] = Sbox[state[i]] ^ k[(r % 2) * 16 + i];
    }
}
void permutation(uint8_t *state){
    int temp[BR];
    for (int j = 0; j < BR; j++){
        temp[j] = state[j];
    }
    for (int j = 0; j < BR; j++){
        state[perm[j]] = temp[j];
    }
}
void enc(uint8_t *m, uint8_t *k, int rounds){
    uint8_t state[BR];
    uint8_t temp[BR_HALF];
    for (int i = 0; i < BR; i++){
        state[i] = m[i];
    }
    for (int i = 0; i < rounds - 1; i++){
        #if PRINT_INTER
        printf("%d round\n", i + 1);
        printState(state);
        #endif
        for (int j = 0; j < BR_HALF; j++){
            temp[j] = state[j * 2];
        }
        /*insert key and Sbox*/
        sboxkey(temp, k, i);
        /*XOR*/
        for (int j = 0; j < BR_HALF; j++){
            state[2 * j + 1] = state[2 * j + 1] ^ temp[j];
        }
        /*add round constants*/
        state[1] = state[1] ^ RC0[i];
        state[3] = state[3] ^ RC1[i];
        /*permutation*/
        permutation(state);
    }
    /*last round function */
    #if PRINT_INTER
    printf("%d round\n", rounds);
    printState(state);
    #endif
    for (int j = 0; j < BR_HALF; j++){
        temp[j] = state[j * 2];
    }
    /*input key and  Sbox*/
    sboxkey(temp, k, rounds - 1);
    /*xor*/
    for (int j = 0; j < BR_HALF; j++){
        state[2 * j + 1] = state[2 * j + 1] ^ temp[j];
    }
    /*add round constants*/
    state[1] = state[1] ^ RC0[rounds-1];
    state[3] = state[3] ^ RC1[rounds-1];
    #if PRINT_INTER
    printState(state);
    #endif
    /*no permutation in the last round*/
    /*copy ciphertext*/
    for (int i = 0; i < BR; i++){
        m[i] = state[i];
    }
}
/* int main(){ */
/*     uint8_t k[BR] = {0x0, 0xa, 0xc, 0xd, 0x0, 0x2, 0x2, 0xf, */
/*                      0x6, 0x8, 0x0, 0xa, 0x5, 0x4, 0x7, 0xf, */
/*                      0xe, 0xe, 0x0, 0x3, 0xc, 0x0, 0x8, 0x6, */
/*                      0x7, 0xb, 0x0, 0x9, 0xe, 0x3, 0xd, 0x7}; */
/*     uint8_t m[BR] = {0xa, 0xf, 0x6, 0xc, 0xd, 0xd, 0x9, 0x0, */
/*                      0xf, 0xc, 0x5, 0xa, 0x6, 0xe, 0xa, 0xa, */
/*                      0x8, 0x9, 0x7, 0xb, 0xc, 0xd, 0x1, 0x2, */
/*                      0x0, 0x8, 0xd, 0x3, 0x9, 0x1, 0xe, 0x1}; */
/*     printf("key :\n"); */
/*     print_state(k); */
/*     printf("plaintext :\n"); */
/*     print_state(m); */
/*     enc(m, k, R); */
/*     printf("ciphertext :\n"); */
/*     print_state(m); */
/*     printf("\n"); */
/*     return 0; */
/* } */