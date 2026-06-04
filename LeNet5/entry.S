.text

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
    li   t0, 0x00006000
    csrs mstatus, t0
    la   t0, trap_handler
    csrw mtvec, t0
    j _start
