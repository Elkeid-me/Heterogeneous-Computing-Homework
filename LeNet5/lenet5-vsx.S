#include "riscv.h"

.globl test_data_index
.globl conv1_kernel_column_first
.globl conv1_kernel_bias
.globl conv1_output

.align 2
.globl lenet5_conv1
.type lenet5_conv1, @function
// a0 is test_data
lenet5_conv1:
    li t0, 784
    li t1, 6
    li t2, 25

    li t3, 0
    j  test_784
loop_784:
    sub t0, t0, t3
    vsetvli t4, t0, e32, m1
    add t0, t0, t3
    li t5, 0
    j test_6
loop_6:
    sub t1, t1, t5
    VSX_SETVSX(__T6, __T1, 8, __IS_FLOAT)
    add t1, t1, t5
    la   a1, conv1_kernel_bias
    slli a2, t3, 2
    add  a1, a1, a2
    li   a2, 3136
    mul  a2, a2, t5
    add  a1, a1, a2

    li   a2, 784
    VSX_LT(__ACC8_0, __A1, __A2)

    li   a3, 0
    j    test_25
loop_25:
    la   a1, test_data_index
    slli a2, t3, 2
    add  a1, a1, a2
    li   a2, 3136
    mul  a2, a2, a3
    add  a1, a1, a2
    VSX_LE32_V(__VSX8, __A1, 0)
    VSX_LGE32_V(__VSX9, __A0, __VSX8)

    la   a1, conv1_kernel_column_first
    slli a2, t3, 2
    add  a1, a1, a2
    li   a2, 24
    mul  a2, a2, a3
    add  a1, a1, a2
    VSX_LETH32_V(__VSX32, __A1, 0)
    VSX_TMULA(__ACC8_0, __VSX9, __VSX32)
    addi a3, a3, 1
test_25:
    blt  a3, t2, loop_25
    la   a1, conv1_output
    slli a2, t3, 2
    add  a1, a1, a2
    li   a2, 3136
    mul  a2, a2, t5
    add  a1, a1, a2
    li   a2, 784
    VSX_ST(__ACC8_0, __A1, __A2)

    add  t5, t5, t6
test_6:
    blt  t5, t1, loop_6

    add  t3, t3, t4
test_784:
    blt  t3, t0, loop_784
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
