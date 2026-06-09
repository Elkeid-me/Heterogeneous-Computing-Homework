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
    for (size_t channel = 0; channel < 6; channel++)
    {
        for (size_t image_i = 0; image_i < 28; image_i++)
        {
            for (size_t image_j = 0; image_j < 28; image_j++)
            {
                float sum = features_0_bias[channel];
                for (size_t kernel_i = 0; kernel_i < 5; kernel_i++)
                {
                    for (size_t kernel_j = 0; kernel_j < 5; kernel_j++)
                    {
                        size_t input_i = image_i + kernel_i;
                        size_t input_j = image_j + kernel_j;
                        sum += features_0_weight[channel * 25 + kernel_i * 5 +
                                                 kernel_j] *
                               test_data[input_i * 32 + input_j];
                    }
                }
                conv1_output[channel * 784 + image_i * 28 + image_j] =
                    (0.0f > sum ? 0.0f : sum);
            }
        }
    }
}
