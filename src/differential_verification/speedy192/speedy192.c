#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#define   NR   7    /* Put the number of rounds here */
#define CYTHON
const uint8_t S[64] = { 0x08,0x00,0x09,0x03,0x38,0x10,0x29,0x13,0x0c,0x0d,0x04,0x07,0x30,0x01,0x20,0x23,
                        0x1a,0x12,0x18,0x32,0x3e,0x16,0x2c,0x36,0x1c,0x1d,0x14,0x37,0x34,0x05,0x24,0x27,
                        0x02,0x06,0x0b,0x0f,0x33,0x17,0x21,0x15,0x0a,0x1b,0x0e,0x1f,0x31,0x11,0x25,0x35,
                        0x22,0x26,0x2a,0x2e,0x3a,0x1e,0x28,0x3c,0x2b,0x3b,0x2f,0x3f,0x39,0x19,0x2d,0x3d };
const uint8_t SI[64] = {0x01,0x0d,0x20,0x03,0x0a,0x1d,0x21,0x0b,0x00,0x02,0x28,0x22,0x08,0x09,0x2a,0x23,
                        0x05,0x2d,0x11,0x07,0x1a,0x27,0x15,0x25,0x12,0x3d,0x10,0x29,0x18,0x19,0x35,0x2b,
                        0x0e,0x26,0x30,0x0f,0x1e,0x2e,0x31,0x1f,0x36,0x06,0x32,0x38,0x16,0x3e,0x33,0x3a,
                        0x0c,0x2c,0x13,0x24,0x1c,0x2f,0x17,0x1b,0x04,0x3c,0x34,0x39,0x37,0x3f,0x14,0x3b };
const uint64_t constants[] = {  0x243f6a8885a308d3,0x13198a2e03707344,0xa4093822299f31d0,0x082efa98ec4e6c89,0x452821e638d01377,0xbe5466cf34e90c6c,
                                0xc0ac29b7c97c50dd,0x3f84d5b5b5470917,0x9216d5d98979fb1b,0xd1310ba698dfb5ac,0x2ffd72dbd01adfb7,0xb8e1afed6a267e96,
                                0xba7c9045f12c7f99,0x24a19947b3916cf7,0x0801f2e2858efc16,0x636920d871574e69,0xa458fea3f4933d7e,0x0d95748f728eb658,
                                0x718bcd5882154aee,0x7b54a41dc25a59b5,0x9c30d5392af26013,0xc5d1b023286085f0,0xca417918b8db38ef,0x8e79dcb0603a180e,
                                0x6c9e0e8bb01e8a3e,0xd71577c1bd314b27,0x78af2fda55605c60,0xe65525f3aa55ab94,0x5748986263e81440,0x55ca396a2aab10b6,
                                0xb4cc5c341141e8ce,0xa15486af7c72e993,0xb3ee1411636fbc2a,0x2ba9c55d741831f6,0xce5c3e169b87931e,0xafd6ba336c24cf5c,
                                0x7a32538128958677,0x3b8f48986b4bb9af,0xc4bfe81b66282193,0x61d809ccfb21a991,0x487cac605dec8032,0xef845d5de98575b1,
                                0xdc262302eb651b88,0x23893e81d396acc5,0x0f6d6ff383f44239,0x2e0b4482a4842004,0x69c8f04a9e1f9b5e,0x21c66842f6e96c9a,
                                0x670c9c61abd388f0,0x6a51a0d2d8542f68,0x960fa728ab5133a3,0x6eef0b6c137a3be4,0xba3bf0507efb2a98,0xa1f1651d39af0176,
                                0x66ca593e82430e88,0x8cee8619456f9fb4,0x7d84a5c33b8b5ebe,0xe06f75d885c12073,0x401a449f56c16aa6,0x4ed3aa62363f7706,
                                0x1bfedf72429b023d,0x37d0d724d00a1248,0xdb0fead349f1c09b,0x075372c980991b7b,0x25d479d8f6e8def7,0xe3fe501ab6794c3b,
                                0x976ce0bd04c006ba,0xc1a94fb6409f60c4,0x5e5c9ec2196a2463,0x68fb6faf3e6c53b5,0x1339b2eb3b52ec6f,0x6dfc511f9b30952c,
                                0xcc814544af5ebd09,0xbee3d004de334afd,0x660f2807192e4bb3,0xc0cba85745c8740f,0xd20b5f39b9d3fbdb,0x5579c0bd1a60320a,
                                0xd6a100c6402c7279,0x679f25fefb1fa3cc,0x8ea5e9f8db3222f8,0x3c7516dffd616b15,0x2f501ec8ad0552ab,0x323db5fafd238760,
                                0x53317b483e00df82,0x9e5c57bbca6f8ca0,0x1a87562edf1769db,0xd542a8f6287effc3,0xac6732c68c4f5573,0x695b27b0bbca58c8,
                                0xe1ffa35db8f011a0,0x10fa3d98fd2183b8,0x4afcb56c2dd1d35b,0x9a53e479b6f84565,0xd28e49bc4bfb9790,0xe1ddf2daa4cb7e33,
                                0x62fb1341cee4c6e8,0xef20cada36774c01,0xd07e9efe2bf11fb4,0x95dbda4dae909198,0xeaad8e716b93d5a0,0xd08ed1d0afc725e0,
                                0x8e3c5b2f8e7594b7,0x8ff6e2fbf2122b64,0x8888b812900df01c,0x4fad5ea0688fc31c,0xd1cff191b3a8c1ad,0x2f2f2218be0e1777,
                                0xea752dfe8b021fa1,0xe5a0cc0fb56f74e8,0x18acf3d6ce89e299,0xb4a84fe0fd13e0b7,0x7cc43b81d2ada8d9,0x165fa26680957705,
                                0x93cc7314211a1477,0xe6ad206577b5fa86,0xc75442f5fb9d35cf,0xebcdaf0c7b3e89a0,0xd6411bd3ae1e7e49,0x00250e2d2071b35e,
                                0x226800bb57b8e0af,0x2464369bf009b91e,0x5563911d59dfa6aa,0x78c14389d95a537f,0x207d5ba202e5b9c5,0x832603766295cfa9,
                                0x11c819684e734a41,0xb3472dca7b14a94a,0x1b5100529a532915,0xd60f573fbc9bc6e4,0x2b60a47681e67400,0x08ba6fb5571be91f,
                                0xf296ec6b2a0dd915,0xb6636521e7b9f9b6,0xff34052ec5855664,0x53b02d5da99f8fa1,0x08ba47996e85076a,0x4b7a70e9b5b32944,
                                0xdb75092ec4192623,0xad6ea6b049a7df7d,0x9cee60b88fedb266,0xecaa8c71699a17ff,0x5664526cc2b19ee1,0x193602a575094c29,
                                0xa0591340e4183a3e,0x3f54989a5b429d65,0x6b8fe4d699f73fd6,0xa1d29c07efe830f5,0x4d2d38e6f0255dc1,0x4cdd20868470eb26,
                                0x6382e9c6021ecc5e,0x09686b3f3ebaefc9,0x3c9718146b6a70a1,0x687f358452a0e286,0xb79c5305aa500737,0x3e07841c7fdeae5c,
                                0x8e7d44ec5716f2b8,0xb03ada37f0500c0d,0xf01c1f040200b3ff,0xae0cf51a3cb574b2,0x25837a58dc0921bd,0xd19113f97ca92ff6,
                                0x9432477322f54701,0x3ae5e58137c2dadc,0xc8b576349af3dda7,0xa94461460fd0030e,0xecc8c73ea4751e41,0xe238cd993bea0e2f,
                                0x3280bba1183eb331,0x4e548b384f6db908,0x6f420d03f60a04bf,0x2cb8129024977c79,0x5679b072bcaf89af,0xde9a771fd9930810,
                                0xb38bae12dccf3f2e,0x5512721f2e6b7124,0x501adde69f84cd87,0x7a5847187408da17,0xbc9f9abce94b7d8c,0xec7aec3adb851dfa,
                                0x63094366c464c3d2,0xef1c18473215d908,0xdd433b3724c2ba16,0x12a14d432a65c451,0x50940002133ae4dd,0x71dff89e10314e55,
                                0x81ac77d65f11199b,0x043556f1d7a3c76b,0x3c11183b5924a509,0xf28fe6ed97f1fbfa,0x9ebabf2c1e153c6e,0x86e34570eae96fb1,
                                0x860e5e0a5a3e2ab3,0x771fe71c4e3d06fa,0x2965dcb999e71d0f,0x803e89d65266c825,0x2e4cc9789c10b36a,0xc6150eba94e2ea78,
                                0xa5fc3c531e0a2df4,0xf2f74ea7361d2b3d   };
typedef bool StateBool[192];
typedef uint8_t StateChar[32];
typedef uint64_t StateUint[3];
StateChar RoundKeys[NR+1];
StateChar RoundCons[NR-1];
void convert_stateuint_to_statechar(StateUint input, StateChar output) {
    for(int i = 0; i < 10; i++)
        output[i] = (input[0] >> (58 - 6 * i)) & 0x3f;
    output[10] = ((input[0] & 0xf) << 2) ^ (input[1] >> 62);
    for(int i = 11; i < 21; i++)
        output[i] = (input[1] >> (122 - 6 * i)) & 0x3f;
    output[21] = ((input[1] & 0x3) << 4) ^ (input[2] >> 60);
    for(int i = 22; i < 32; i++)
        output[i] = (input[2] >> (186 - 6 * i)) & 0x3f;
}
void convert_statechar_to_stateuint(StateChar input, StateUint output) {
    output[0] = 0;
    for(int i = 0; i < 10; i++){
        output[0] <<= 6;
        output[0] ^= (input[i] & 0x3f);
    }
    output[0] <<= 4;
    output[0] ^= (input[10] >> 2) & 0xf;
    output[1] = 0;
    output[1] ^= (input[10] & 0x3);
    for(int i = 11; i < 21; i++){
        output[1] <<= 6;
        output[1] ^= (input[i] & 0x3f);
    }
    output[1] <<= 2;
    output[1] ^= (input[21] >> 4) & 0x3;
    output[2] = 0;
    output[2] ^= (input[21] & 0xf);
    for(int i = 22; i < 32; i++){
        output[2] <<= 6;
        output[2] ^= (input[i] & 0x3f);
    }
}
void convert_statebool_to_statechar(StateBool input, StateChar output) {
    for(int i = 0; i < 32; i++) {
        output[i] = 0;
        for(int j = 0; j < 6; j++) {
            output[i] <<= 1;
            output[i] ^= input[6 * i + j];
        }
    }
}
void print_state(StateChar input) {
    for(int i = 0; i < 16; i++)
        printf("%X%X%X", input[2 * i] >> 2, ((input[2 * i] & 3) << 2) ^ (input[2 * i + 1] >> 4), input[2 * i + 1] & 0xf);
    printf("\n");
}
void print_state64(StateUint input){
    for(int i=0; i<3; i++){
        printf("%016llX, ", input[i]);
    }
    printf("\n");
}
void print_state6(StateChar input){
    for(int i=0; i<32; i++){
        printf("0x%02x,", input[i]);
    }
    printf("\n");
}
void prepare_round_cons() {
    StateUint RCuint;
    /* printf("----------------------------------------------------\n"); */
    for(int r = 0; r < NR - 1; r++) {
        RCuint[0] = constants[3 * r];
        RCuint[1] = constants[3 * r + 1];
        RCuint[2] = constants[3 * r + 2];
        convert_stateuint_to_statechar(RCuint, RoundCons[r]);
        /* print_state6(RoundCons[r]); */
    }
    /* printf("----------------------------------------------------\n"); */
}
void prepare_round_keys(StateChar Key) {
    StateBool TempRoundKeyState[2];
    for(int i = 0; i < 32; i++)
        for(int j = 0; j < 6; j++)
            TempRoundKeyState[0][6 * i + j] = (Key[i] >> (5 - j)) & 1;
    convert_statebool_to_statechar(TempRoundKeyState[0], RoundKeys[0]);
    for(int r = 1; r <= NR; r++) {
        bool ind_new = (r % 2);
        bool ind_old = !ind_new;
        /* printf("%d %d \n", ind_new, ind_old); */
        for(int i = 0; i < 192; i++)
            TempRoundKeyState[ind_new][i] = TempRoundKeyState[ind_old][(7 * i + 1) % 192];
        convert_statebool_to_statechar(TempRoundKeyState[ind_new], RoundKeys[r]);
    }
}
void AK(StateChar input, int r) {
    for(int i = 0; i < 32; i++)
        input[i] ^= RoundKeys[r][i];
}
void AC(StateChar input, int r) {
    for(int i = 0; i < 32; i++)
        input[i] ^= RoundCons[r][i];
}
void SB(StateChar input) {
    for(int i = 0; i < 32; i++)
        input[i] = S[(int)input[i]];
}
void MC(StateChar input) {
    const int alphas[] = {1, 5, 9, 15, 21, 26};
    StateChar temp;
    for(int i = 0; i < 32; i++)
        temp[i] = input[i];
    for(int a = 0; a < (sizeof(alphas) / sizeof(alphas[0])); a++)
    {
        for(int i = 0; i < 32; i++)
            input[i] ^= temp[(i + alphas[a]) % 32];
    }
}
void SC(StateChar input) {
    bool temp[32][6];
    for(int i = 0; i < 32; i++)
        for(int j = 0; j < 6; j++)
            temp[i][j] = ((input[(i + j) % 32] >> (5 - j)) & 1);
    for(int i = 0; i < 32; i++) {
        input[i] = 0;
        for(int j = 0; j < 6; j++) {
            input[i] <<= 1;
            input[i] ^= temp[i][j];
        }
    }
}
void Encrypt(uint8_t *plaintext, uint8_t *key, int rounds) {
    prepare_round_cons();
    prepare_round_keys(key);
    for(int r = 0; r < rounds; r++) {
     /* print_state6(plaintext); */
        AK(plaintext, r);
     /* print_state6(plaintext); */
        SB(plaintext);
     /* print_state6(plaintext); */
        SC(plaintext);
     /* print_state6(plaintext); */
        SB(plaintext);
     /* print_state6(plaintext); */
        if (r == (rounds - 1))
            AK(plaintext, rounds);
        else {
            SC(plaintext);
            MC(plaintext);
     /* print_state6(plaintext); */
            AC(plaintext, r);
        }
    }
}
void SBI(StateChar input) {
    for(int i = 0; i < 32; i++)
        input[i] = SI[(int)input[i]];
}
void MCI(StateChar input) {
    const int alphas[] = {4, 5, 6, 7, 10, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 28};
    StateChar temp;
    for(int i = 0; i < 32; i++)
        temp[i] = input[i];
    for(int a = 0; a < (sizeof(alphas) / sizeof(alphas[0])); a++)
    {
        for(int i = 0; i < 32; i++)
            input[i] ^= temp[(i + alphas[a]) % 32];
    }
}
void SCI(StateChar input) {
    bool temp[32][6];
    for(int i = 0; i < 32; i++)
        for(int j = 0; j < 6; j++)
            temp[(i + j) % 32][j] = ((input[i] >> (5 - j)) & 1);
    for(int i = 0; i < 32; i++) {
        input[i] = 0;
        for(int j = 0; j < 6; j++) {
            input[i] <<= 1;
            input[i] ^= temp[i][j];
        }
    }
}
void Decrypt(uint8_t *Ciphertext, uint8_t *Key, int rounds) {
    prepare_round_cons();
    prepare_round_keys(Key);
    AK (Ciphertext, rounds);
    SBI(Ciphertext);
    SCI(Ciphertext);
    SBI(Ciphertext);
    AK (Ciphertext, rounds - 1);
    for(int r = rounds - 2; r >= 0; r--) {
        AC (Ciphertext, r);
        MCI(Ciphertext);
        SCI(Ciphertext);
        SBI(Ciphertext);
        SCI(Ciphertext);
        SBI(Ciphertext);
        AK (Ciphertext, r);
    }
}
/* #include "diff.h" */
void test_convert(){
    StateUint plaintext     = {0xA13A632451070E43,0x82A27F26A40682F3,0xFE9FF68028D24FDB};
    StateChar plaintextchar;
    print_state64(plaintext);
    convert_stateuint_to_statechar(plaintext, plaintextchar);
    print_state6(plaintextchar);
    convert_statechar_to_stateuint(plaintextchar, plaintext);
    print_state64(plaintext);
}
#ifndef CYTHON
int main() {
    /* test_convert(); */
    StateUint Plaintext     = {0xA13A632451070E43,0x82A27F26A40682F3,0xFE9FF68028D24FDB};
    StateUint Key           = {0x764C4F6254E1BFF2,0x08E95862428FAED0,0x1584F4207A7E8477};
    StateUint Ciphertext    = {0xED3D0EA11C427BD3,0x2570DF41C6FD66EB,0xBF4916E760ED0943};
    StateChar plaintextchar, keychar;
    convert_stateuint_to_statechar(Plaintext, plaintextchar);
    convert_stateuint_to_statechar(Key, keychar);
    print_state(plaintextchar);
    print_state6(plaintextchar);
    print_state6(keychar);
    Encrypt(plaintextchar, keychar, NR);
    print_state(plaintextchar);
    print_state6(plaintextchar);
    print_state6(keychar);
    Decrypt(plaintextchar, keychar, NR);
    print_state(plaintextchar);
    print_state6(plaintextchar);
    print_state6(keychar);
    /* test_diff(); */
    /* generate_naive_diff(2); */
    return 0;
}
#endif