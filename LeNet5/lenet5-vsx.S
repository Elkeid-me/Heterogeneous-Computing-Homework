#include "riscv.h"

.globl test_data_index
.globl conv1_kernel_column_first
.globl conv1_kernel_bias
.globl conv1_output

.align 2
.globl lenet5_conv1
.type lenet5_conv1, @function
lenet5_conv1:
    li t0, 784
    li t2, 25   // const
    li t3, 784  // const
    la t4, conv1_kernel_bias
    la t5, test_data_index
    la t6, conv1_kernel_column_first
    la a1, conv1_output
    j  test_784
loop_784:
    vsetvli a2, t0, e32, m1
    li t1, 6
    j test_6
loop_6:
    VSX_SETVSX(__A3, __T1, 8, __IS_FLOAT)

    VSX_LT(__ACC8_0, __T4, __T3)
    li  a4, 0
    j   test_25
loop_25:
    li  a6, 3136
    mul a6, a4, a6
    mv  a5, t5
    add a6, a5, a6
    VSX_LE32_V(__VSX8, __A6, 0)
    VSX_LGE32_V(__VSX9, __A0, __VSX8)
    li  a6, 24
    mul a6, a6, a4
    mv  a5, t6
    add a6, a5, a6
    VSX_LETH32_V(__VSX32, __A6, 0)
    VSX_TMULA(__ACC8_0, __VSX9, __VSX32)
    addi a4, a4, 1
test_25:
    blt  a4, t2, loop_25
    VSX_ST(__ACC8_0, __A1, __T3)

    sub  t1, t1, a3
    slli a3, a3, 2
    add  t6, t6, a3
    mul  a3, a3, t3
    add  t4, t4, a3
    add  a1, a1, a3
test_6:
    bgtz t1, loop_6

    sub  t0, t0, a2
    slli a2, a2, 2
    add  t4, t4, a2
    add  t5, t5, a2
    add  a1, a1, a2

    la  t6, conv1_kernel_column_first
    li  a4, 18816
    sub t4, t4, a4
    sub a1, a1, a4
test_784:
    bgtz t0, loop_784
    ret

.align 2
.globl trap_handler
.type trap_handler, @function
trap_handler:
    addi  sp, sp, -8 * 32
    sd    x1, 0x0(sp)
    sd    x4, 3 * 8(sp)
    sd    x5, 4 * 8(sp)
    sd    x6, 5 * 8(sp)
    sd    x7, 6 * 8(sp)
    sd    x10, 9 * 8(sp)
    sd    x11, 10 * 8(sp)
    sd    x12, 11 * 8(sp)
    sd    x13, 12 * 8(sp)
    sd    x14, 13 * 8(sp)
    sd    x15, 14 * 8(sp)
    sd    x16, 15 * 8(sp)
    sd    x17, 16 * 8(sp)
    sd    x28, 27 * 8(sp)
    sd    x29, 28 * 8(sp)
    sd    x30, 29 * 8(sp)
    sd    x31, 30 * 8(sp)
    csrr  a0, mcause
    csrr  a1, mepc
    csrr  a2, mtval
    jal   ra, c_trap_handler
    ld    x1, 0x0(sp)
    ld    x4, 3 * 8(sp)
    ld    x5, 4 * 8(sp)
    ld    x6, 5 * 8(sp)
    ld    x7, 6 * 8(sp)
    ld    x10, 9 * 8(sp)
    ld    x11, 10 * 8(sp)
    ld    x12, 11 * 8(sp)
    ld    x13, 12 * 8(sp)
    ld    x14, 13 * 8(sp)
    ld    x15, 14 * 8(sp)
    ld    x16, 15 * 8(sp)
    ld    x17, 16 * 8(sp)
    ld    x28, 27 * 8(sp)
    ld    x29, 28 * 8(sp)
    ld    x30, 29 * 8(sp)
    ld    x31, 30 * 8(sp)
    addi  sp, sp, 8 * 32
    mret

.align 2
.globl _reset_vector
.type _reset_vector, @function
_reset_vector:
    li   t0, 0x00006c00
    csrs mstatus, t0
    la   t0, trap_handler
    csrw mtvec, t0
    j _start
