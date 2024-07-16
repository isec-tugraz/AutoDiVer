#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define ROUNDS 20
void printState(uint8_t *S) {
    for(int i=0; i<16; i++) {
        printf("%02X", S[i]);
    }
    printf("\n");
}
void Copy(uint8_t *T, uint8_t *S) {
    for(int i=0; i<16; i++) {
        T[i] = S[i];
    }
}

void unpackBits(uint8_t *cellBin, uint8_t cell){
    for(int j=0; j<8; j++) {
        cellBin[7 - j] = (cell >> j) & 0x01;
    }
}
uint8_t packBits(uint8_t *cellBin){
    uint8_t cell = 0;
    for(int j=0; j<8; j++) {
        cell = (cell << 1) | cellBin[j];
    }
    return cell;
}

void printBits(uint8_t *cell) {
    for(int i=0; i<8; i++) {
        printf("%d", cell[i]);
    }
    printf("\n");
}
uint8_t prePermute(uint8_t cell, int i) {
    uint8_t perm0[8] = {4,1,6,3,0,5,2,7};
    uint8_t perm1[8] = {1,6,7,0,5,2,3,4};
    uint8_t perm2[8] = {2,3,4,1,6,7,0,5};
    uint8_t perm3[8] = {7,4,1,2,3,0,5,6};

    uint8_t cellBin[8];
    uint8_t cellBinP[8];
    unpackBits(cellBin, cell);

    /* printBits(cellBin); */
    for(int j=0; j<8; j++){
        if(i == 0)
            cellBinP[j] = cellBin[perm0[j]];
        if(i == 1)
            cellBinP[j] = cellBin[perm1[j]];
        if(i == 2)
            cellBinP[j] = cellBin[perm2[j]];
        if(i == 3)
            cellBinP[j] = cellBin[perm3[j]];
    }
    /* printBits(cellBinP); */

    cell = packBits(cellBinP);
    return cell;
}
uint8_t postPermute(uint8_t cell, int i) {
    uint8_t perm0[8] = {4,1,6,3,0,5,2,7};
    uint8_t perm1[8] = {1,6,7,0,5,2,3,4};
    uint8_t perm2[8] = {2,3,4,1,6,7,0,5};
    uint8_t perm3[8] = {7,4,1,2,3,0,5,6};

    uint8_t cellBin[8];
    uint8_t cellBinP[8];
    unpackBits(cellBinP, cell);
    for(int j=0; j<8; j++){
        if(i == 0)
            cellBin[perm0[j]] = cellBinP[j];
        if(i == 1)
            cellBin[perm1[j]] = cellBinP[j];
        if(i == 2)
            cellBin[perm2[j]] = cellBinP[j];
        if(i == 3)
            cellBin[perm3[j]] = cellBinP[j];
    }
    cell = packBits(cellBin);

    return cell;
}

uint8_t SSBi(uint8_t cell, int i) {
    uint8_t SB1[16] = {0x1, 0x0, 0x5, 0x3, 0xe, 0x2, 0xf, 0x7,
                       0xd, 0xa, 0x9, 0xb, 0xc, 0x8, 0x4, 0x6};
    cell = prePermute(cell, i);
    cell = (SB1[(cell >> 4) & 0x0F] << 4 ) | SB1[cell & 0x0F];
    cell = postPermute(cell, i);
    return cell;
}

void sbox(uint8_t *msg){
    for(int8_t i = 0; i < 16; i++){
        msg[i] = SSBi(msg[i], i%4);
    }
}

void sr(uint8_t *msg){
    uint8_t sr_index[16] = {0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8};
    uint8_t temp[16];
    Copy(temp, msg);

    /* for each nibble */
    for(int i=0;i<16;i++){
        msg[i] = temp[sr_index[i]];
    }
}

void mc(uint8_t *msg){
    uint8_t nibble[16];

    for (uint8_t col=0; col<4; col++){
        nibble[0] = msg[(4*col) + 0];
        nibble[1] = msg[(4*col) + 1];
        nibble[2] = msg[(4*col) + 2];
        nibble[3] = msg[(4*col) + 3];

        msg[(4*col) + 0] = nibble[1]^nibble[2]^nibble[3];
        msg[(4*col) + 1] = nibble[0]^nibble[2]^nibble[3];
        msg[(4*col) + 2] = nibble[0]^nibble[1]^nibble[3];
        msg[(4*col) + 3] = nibble[0]^nibble[1]^nibble[2];
    }
}


void add_round_key(uint8_t *msg, uint8_t *key){
    for (int i = 0; i < 16; i++) {
        msg[i] = msg[i] ^ key[i];
    }
}

void enc_midori128(uint8_t *cip, uint8_t *key, int rounds){
    uint8_t beta[][16] = {
        {0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1},
        {0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
        {1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1},
        {0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1},
        {1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0},
        {0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0},
        {1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0},
        {0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1},
        {0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0},
        {0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0},
        {1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0},
        {0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0}
    };

    if (rounds > ROUNDS){
        fprintf(stderr, "Number of rounds is too high\n");
        abort();
    }

    add_round_key(cip, key);
    /* printState(cip); */

    for (uint8_t i = 0; i < rounds - 1; i++){
      if (i >= sizeof(beta) / sizeof(beta[0]))
      {
        fprintf(stderr, "array index out of bounds\n");
        abort();
      }
    /* for (uint8_t i = 0; i < rounds; i++){ */
    /* printf("------------------------------------------\n"); */
        sbox(cip);
    /* printState(cip); */
        sr(cip);
    /* printState(cip); */
        mc(cip);
    /* printState(cip); */
        add_round_key(cip, key);
        add_round_key(cip, beta[i]);
    /* printState(cip); */
    /* printf("------------------------------------------\n"); */
    }
    sbox(cip);
    add_round_key(cip, key);
    /* printState(cip); */
    /* printf("------------------------------------------\n"); */
}

/* void testSbox() { */
/*     for(int j=0; j<16; j++) { */
/*         printf("%02X ", SSBi(j, 0)); */
/*     } */
/*     printf("\n"); */
/* } */

/* int main(){ */
/*     uint8_t msg[16] = {0x51, 0x08, 0x4C, 0xE6, 0xE7, 0x3A, 0x5C, 0xA2, */
/*                        0xEC, 0x87, 0xD7, 0xBA, 0xBC, 0x29, 0x75, 0x43}; */
/*     uint8_t cip[16] = {0}; */
/*     uint8_t key[16] = {0x68, 0x7d, 0xed, 0x3b, 0x3c, 0x85, 0xb3, 0xf3, */
/*                        0x5b, 0x10, 0x09, 0x86, 0x3e, 0x2a, 0x8c, 0xbf}; */
/*     printState(msg); */
/*     enc_midori128(msg, key, ROUNDS); */
/*     printState(key); */
/*     printState(msg); */

/*     /1* testSbox(); *1/ */
/* } */
