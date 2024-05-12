#include <stdio.h>

// 1 zadani
__uint8_t data[4] = {1,2,3,4}; //global
extern void swap_endiannes();

// 2 zadani
__uint8_t data2[4] =  {0xAF, 0xBE, 0xAD, 0xDE}; // global
int result;
extern void compose();

// 3 zadani
char data3[10] = "XXX0000"; // global
extern void replace();

// 4 zadani
__int16_t key = -96; // global
__int64_t extended;
extern void extend();

int main () {
    // 1 zadani
    swap_endiannes();
    printf(
    "Array %d, %d, %d, %d\n", 
    data[0],
    data[1],
    data[2],
    data[3]
    );

    // 2 zadani
    compose();
    printf(
    "result %u \n", 
    result
    );

    // 3 zadani
    replace();
    printf(
    "login %s \n", 
    data3
    );

    // 4 zadani
    extend();
    printf(
    "extended %ld \n", 
    extended
    );

    return 0;
}