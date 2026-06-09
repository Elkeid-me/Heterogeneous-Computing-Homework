#include "lenet5_weights.h"
#include <stdio.h>

float conv1_output[4704];

void c_trap_handler(int cause, int epc, int tval)
{
    static int count = 0;
    if (count < 10)
    {
        printf("trap: cause=%x, epc=%x, tval=%x\n", cause, epc, tval);
        count++;
    }
}

void lenet5_conv1(const float *);

int main(void)
{
    lenet5_conv1(test_data);
}
