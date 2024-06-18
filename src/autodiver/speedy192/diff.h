#include "ddt.h"
void get_diff(uint8_t *diff, uint8_t *x, uint8_t *y){
    for(int i=0; i<32; i++) {
        diff[i] = x[i] ^ y[i];
    }
}
void constructDiff(uint8_t *p1, uint8_t *p2, uint8_t *key, int rounds) {
    prepare_round_cons();
    prepare_round_keys(key);
    uint8_t diff[32];
    for(int r = 0; r < rounds; r++) {
        AK(p1, r);
        AK(p2, r);
        /*----------------------------------*/
        get_diff(diff, p1, p2);
        printf("indiff %d0:", r);
        print_state6(diff);
        /*-----------------------------------*/
        SB(p1);
        SB(p2);
        /*----------------------------------*/
        get_diff(diff, p1, p2);
        printf("oudiff %d0:", r);
        print_state6(diff);
        /*-----------------------------------*/
        SC(p1);
        SC(p2);
        /*----------------------------------*/
        get_diff(diff, p1, p2);
        printf("indiff %d0:", r);
        print_state6(diff);
        /*-----------------------------------*/
        SB(p1);
        SB(p2);
        /*----------------------------------*/
        get_diff(diff, p1, p2);
        printf("oudiff %d0:", r);
        print_state6(diff);
        /*-----------------------------------*/
        if (r == (rounds - 1)){
            AK(p1, rounds);
            AK(p2, rounds);
        }
        else {
            SC(p1);
            SC(p2);
            MC(p1);
            MC(p2);
            AC(p1, r);
            AC(p2, r);
        }
    }
}
void test_diff(){
    StateChar p1char, p2char, keychar;
    StateUint Key  = {0x764C4F6254E1BFF2,0x08E95862428FAED0,0x1584F4207A7E8477};
    StateUint p1   = {0xA13A632451070E43,0x82A27F26A40682F3,0xFE9FF68028D24FDB};
    StateUint p2   = {0x013A632451070E43,0x82A27F26A40682F3,0xFE9FF68028D24FDB};
    convert_stateuint_to_statechar(p1, p1char);
    convert_stateuint_to_statechar(p2, p2char);
    convert_stateuint_to_statechar(Key, keychar);
    print_state6(p1char);
    print_state6(p2char);
    print_state6(keychar);
    constructDiff(p1char, p2char, keychar, 2);
    print_state6(p1char);
    print_state6(p2char);
    print_state6(keychar);
}
uint8_t get_best_output_diff(uint8_t indiff){
    uint8_t max_value = DDT[indiff][0];
    uint8_t max_index = 0;
    for(int i=1; i<64; i++){
        if(DDT[indiff][i] > max_value){
            max_value = DDT[indiff][i];
            max_index = i;
        }
    }
    return max_index;
}
void create_diff_from_ddt(uint8_t *diff){
    for(int i=0; i<32; i++){
        diff[i] = get_best_output_diff(diff[i]);
    }
}
void generate_naive_diff(int rounds){
    StateChar diff = {01};
    for(int r = 0; r < rounds; r++) {
        printf("((");
        print_state6(diff);
        printf("),\n");
        create_diff_from_ddt(diff);
        printf("(");
        print_state6(diff);
        printf(")),\n\n");
        SC(diff);
        printf("((");
        print_state6(diff);
        printf("),");
        create_diff_from_ddt(diff);
        printf("(");
        print_state6(diff);
        printf(")),\n\n");
        if (r == (rounds - 1)){
            continue;
        }
        else {
            SC(diff);
            MC(diff);
        }
    }
}