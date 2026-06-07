#include "lenet5_weights.h"
#include <stdio.h>

void c_trap_handler(int cause, int epc, int tval)
{
    static int count = 0;
    if (count < 10)
    {
        printf("trap: cause=%x, epc=%x, tval=%x\n", cause, epc, tval);
        count++;
    }
}

// int lenet5(const float *input);

void lenet5_conv1(const float *);

int main(void)
{
    lenet5_conv1(test_data);
    for (size_t i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < 784; j++)
            printf("%f, ", conv1_output[i * 784 + j]);
        printf("\n");
    }
}
