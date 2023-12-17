#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
void printreg_to_file(const void *a, int nrof_byte, FILE *fp){
    int i;
    unsigned char *f = (unsigned char *)a;
    for(i=0; i < nrof_byte; i++){
        fprintf(fp, "%02X ",(unsigned char) f[nrof_byte - 1 - i]); //uint8_t c[4+8];
    }
    fprintf(fp, "\n");
}
void printreg(const void *a, int nrof_byte){
    printreg_to_file(a, nrof_byte, stdout);
}
#define ROUNDS 28
#define NR_OF_SBOX 16
#define NIBBLE 4
#define NR_OF_BYTE 8
#define GRP 4
int rc[28] = {0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B};
int SB_table[16] = {0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe};
int inv_SB_table[16] = {0xd, 0x0, 0x8, 0x6, 0x2, 0xc, 0x4, 0xb, 0xe, 0x7, 0x1, 0xa, 0x3, 0x9, 0xf, 0x5};
int p_layer_table[64] = {   0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3,														  4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7,														8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,													     12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15};
int inv_p_layer_table[64] = {	0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63, 													     12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59, 													    8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55, 													  4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51};
uint64_t circ_right_shift(uint64_t k, int n){
    uint64_t temp=(((k >> n) & 0xFFFF)|((k << (16 - n) )&0xFFFF))&0xFFFF;
    return temp;
}
void rotate_word(uint64_t *round_key_word){
    uint64_t temp1 = round_key_word[0];
    uint64_t temp2 = round_key_word[1];
    for(int i=2; i<8; i++){
        round_key_word[i-2] = round_key_word[i];
    }
    round_key_word[6] = temp1;
    round_key_word[7] = temp2;
}
void generate_round_keys(uint64_t key[2], uint64_t *round_keys){
    memset(round_keys, 0x00, ROUNDS*sizeof(uint64_t));
    uint64_t round_key_word[8];
    for(int i=0; i<4; i++){
        round_key_word[i] = (key[0] >> 16*i) & 0xFFFF;
        round_key_word[4 + i] = (key[1] >> 16*i) & 0xFFFF;
    }
    for(int r=0; r<ROUNDS; r++){
        round_keys[r] = (round_key_word[1] << 16) | round_key_word[0];
        rotate_word(round_key_word);
        round_key_word[6] = circ_right_shift(round_key_word[6], 12);
        round_key_word[7] = circ_right_shift(round_key_word[7],  2);
    }
}
uint64_t SBox(uint64_t msg){
    uint64_t cip = 0UL;
    uint64_t nibble;
    for(int i=0; i<NR_OF_SBOX; i++){
        nibble = SB_table[(msg >> 4*i)&0xf];
        cip = cip| (nibble << (4*i));
    }
    return cip;
}
uint64_t invSBox(uint64_t msg){
    uint64_t cip = 0UL;
    uint64_t nibble;
    for(int i=0; i<NR_OF_SBOX; i++){
        nibble = inv_SB_table[(msg >> 4*i)&0xf];
        cip = cip| (nibble << (4*i));
    }
    return cip;
}
uint64_t PLayer(uint64_t msg){
    uint64_t cip = 0UL;
    for(int i=0; i<64; i++){
        cip = (cip << 1)|((msg >> inv_p_layer_table[63-i])&1);
    }
    return cip;
}
uint64_t invPLayer(uint64_t msg){
    uint64_t cip = 0UL;
    for(int i=0; i<64; i++){
        cip = cip|(((msg>>i)&1) << inv_p_layer_table[i]);
    }
    return cip;
}
uint64_t addRk(uint64_t msg, uint64_t key){
    uint64_t U,V, U1=0, V1=0;
    V = key&0xFFFF;
    U = (key>>16)&0xFFFF;
    for (int i=0; i<16; i++){
        V1 = (V1 << 4) | ((V >> (15-i)) & 0x01);
        U1 = (U1 << 4) | ((U >> (15-i)) & 0x01);
    }
    U1 = U1 << 1;
    msg = msg ^ U1 ^ V1;
    return msg;
}
uint64_t addRc(uint64_t msg, int r){
    uint64_t rcon = 0UL;
    int rcon_pos[6] = {23, 19, 15, 11, 7, 3};
	for(int i=0;i<6;i++){
		rcon = rcon | (((rc[r]>>i)&1) << rcon_pos[5-i]);
		}
	rcon = rcon|0x8000000000000000;
	return (msg^rcon);
}
uint64_t enc_gift64(uint64_t msg, uint64_t key[2]){
    uint64_t round_keys[ROUNDS];
    //printreg(key, 16);
    generate_round_keys(key, round_keys);
    for(int r=0; r<ROUNDS; r++){
        msg = SBox(msg);
        msg = PLayer(msg);
        msg = addRk(msg, round_keys[r]);
        msg = addRc(msg, r);
    }
    return msg;
}
uint64_t dec_gift64(uint64_t msg, uint64_t key[2]){
    uint64_t round_keys[ROUNDS];
    generate_round_keys(key, round_keys);
    for(int r=ROUNDS-1; r>=0; r--){
        if(r !=(ROUNDS-1)){
            msg = addRc(msg, r);
        }
        //printf("M[%d]: %016lX\n\n", r, msg);
        msg = addRk(msg, round_keys[r]);
        //printf("M[%d]: %016lX\n\n", r, msg);
        if(r !=(ROUNDS-1)){
            msg = invPLayer(msg);
        }
        //printf("M[%d]: %016lX\n\n", r, msg);
        msg = invSBox(msg);
        //printf("M[%d]: %016lX\n\n", r, msg);
    }
    return msg;
}
uint64_t rand64(){
    uint64_t random64 = ((uint64_t)rand() << 32) | rand();;
    return random64;
}
void test_gift_64(){
    uint64_t msg, cip, key[2], dip;
    msg = rand64();
    key[0] = rand64();
    key[1] = rand64();
    printreg(key, 16);
    cip = enc_gift64(msg, key);
    printreg(&msg, 8);
    printreg(&cip, 8);
    printreg(key, 16);
    dip = dec_gift64(cip, key);
    printreg(&dip, 8);
    printreg(key, 16);
}
int main(){
    srand(time(NULL));
    test_gift_64();
    return 0;
}