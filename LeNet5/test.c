#include <stdio.h>
#include <stddef.h>

int *test();

void c_trap_handler(int cause, int epc, int tval)
{
    static int count = 0;
    if (count < 10)
    {
        printf("trap: cause=%x, epc=%x, tval=%x\n", cause, epc, tval);
        count++;
    }
}

int main()
{
    int *ptr = test();
    for (size_t i = 0; i < 16; i++)
    {
        for (size_t j = 0; j < 16; j++)
            printf("%3d, ", ptr[i * 16 + j]);
        printf("\n");
    }
    return 0;
}
