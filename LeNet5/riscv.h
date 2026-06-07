#define __X0 0
#define __X1 1
#define __X2 2
#define __X3 3
#define __X4 4
#define __X5 5
#define __X6 6
#define __X7 7
#define __X8 8
#define __X9 9
#define __X10 10
#define __X11 11
#define __X12 12
#define __X13 13
#define __X14 14
#define __X15 15
#define __X16 16
#define __X17 17
#define __X18 18
#define __X19 19
#define __X20 20
#define __X21 21
#define __X22 22
#define __X23 23
#define __X24 24
#define __X25 25
#define __X26 26
#define __X27 27
#define __X28 28
#define __X29 29
#define __X30 30
#define __X31 31

#define __ZERO __X0
#define __RA __X1
#define __SP __X2
#define __GP __X3
#define __TP __X4
#define __T0 __X5
#define __T1 __X6
#define __T2 __X7
#define __S0 __X8
#define __FP __X8
#define __S1 __X9
#define __A0 __X10
#define __A1 __X11
#define __A2 __X12
#define __A3 __X13
#define __A4 __X14
#define __A5 __X15
#define __A6 __X16
#define __A7 __X17
#define __S2 __X18
#define __S3 __X19
#define __S4 __X20
#define __S5 __X21
#define __S6 __X22
#define __S7 __X23
#define __S8 __X24
#define __S9 __X25
#define __S10 __X26
#define __S11 __X27
#define __T3 __X28
#define __T4 __X29
#define __T5 __X30
#define __T6 __X31

#define __F0 0
#define __F1 1
#define __F2 2
#define __F3 3
#define __F4 4
#define __F5 5
#define __F6 6
#define __F7 7
#define __F8 8
#define __F9 9
#define __F10 10
#define __F11 11
#define __F12 12
#define __F13 13
#define __F14 14
#define __F15 15
#define __F16 16
#define __F17 17
#define __F18 18
#define __F19 19
#define __F20 20
#define __F21 21
#define __F22 22
#define __F23 23
#define __F24 24
#define __F25 25
#define __F26 26
#define __F27 27
#define __F28 28
#define __F29 29
#define __F30 30
#define __F31 31

#define __V0 0
#define __V1 1
#define __V2 2
#define __V3 3
#define __V4 4
#define __V5 5
#define __V6 6
#define __V7 7
#define __V8 8
#define __V9 9
#define __V10 10
#define __V11 11
#define __V12 12
#define __V13 13
#define __V14 14
#define __V15 15
#define __V16 16
#define __V17 17
#define __V18 18
#define __V19 19
#define __V20 20
#define __V21 21
#define __V22 22
#define __V23 23
#define __V24 24
#define __V25 25
#define __V26 26
#define __V27 27
#define __V28 28
#define __V29 29
#define __V30 30
#define __V31 31

#define __VSX0 0
#define __VSX1 1
#define __VSX2 2
#define __VSX3 3
#define __VSX4 4
#define __VSX5 5
#define __VSX6 6
#define __VSX7 7
#define __VSX8 8
#define __VSX9 9
#define __VSX10 10
#define __VSX11 11
#define __VSX12 12
#define __VSX13 13
#define __VSX14 14
#define __VSX15 15
#define __VSX16 16
#define __VSX17 17
#define __VSX18 18
#define __VSX19 19
#define __VSX20 20
#define __VSX21 21
#define __VSX22 22
#define __VSX23 23
#define __VSX24 24
#define __VSX25 25
#define __VSX26 26
#define __VSX27 27
#define __VSX28 28
#define __VSX29 29
#define __VSX30 30
#define __VSX31 31
#define __VSX32 32
#define __VSX33 33
#define __VSX34 34
#define __VSX35 35
#define __VSX36 36
#define __VSX37 37
#define __VSX38 38
#define __VSX39 39
#define __VSX40 40
#define __VSX41 41
#define __VSX42 42
#define __VSX43 43
#define __VSX44 44
#define __VSX45 45
#define __VSX46 46
#define __VSX47 47
#define __VSX48 48
#define __VSX49 49
#define __VSX50 50
#define __VSX51 51
#define __VSX52 52
#define __VSX53 53
#define __VSX54 54
#define __VSX55 55
#define __VSX56 56
#define __VSX57 57
#define __VSX58 58
#define __VSX59 59
#define __VSX60 60
#define __VSX61 61
#define __VSX62 62
#define __VSX63 63

#define __ACC4_0 0
#define __ACC4_1 1
#define __ACC4_2 2
#define __ACC4_3 3
#define __ACC4_4 4
#define __ACC4_5 5
#define __ACC4_6 6
#define __ACC4_7 7
#define __ACC4_8 8
#define __ACC4_9 9
#define __ACC4_10 10
#define __ACC4_11 11
#define __ACC4_12 12
#define __ACC4_13 13
#define __ACC4_14 14
#define __ACC4_15 15

#define __ACC8_0 16
#define __ACC8_1 17
#define __ACC8_2 18
#define __ACC8_3 19
#define __ACC8_4 20
#define __ACC8_5 21
#define __ACC8_6 22
#define __ACC8_7 23

#define __ACC16_0 24
#define __ACC16_1 25
#define __ACC16_2 26
#define __ACC16_3 27

#define __ACC32_0 28
#define __ACC32_1 29

#define __IS_INT 0
#define __IS_FLOAT 1

// 0xA is Custom-0
#define VSX_R_TYPE(VD, VS1, VS2, FUNCT2, FUNCT5)                               \
    .word(11 | ((VD & 0x3F) << 7) | ((VS1 & 0x3F) << 13) |                     \
          ((VS2 & 0x3F) << 19) | (FUNCT2 << 25) | (FUNCT5 << 27))

#define VSX_I_TYPE(VD, VS1, IMM, FUNCT5)                                       \
    .word(11 | ((VD & 0x3F) << 7) | ((VS1 & 0x3F) << 13) | (IMM << 19) |       \
          (FUNCT5 << 27))

#define VSX_LE8_V(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 0)
#define VSX_LE16_V(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 1)
#define VSX_LE32_V(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 2)
#define VSX_LE64_V(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 3)
#define VSX_SE8_V(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 0)
#define VSX_SE16_V(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 1)
#define VSX_SE32_V(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 2)
#define VSX_SE64_V(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 3)

#define VSX_LETH8_V(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 4)
#define VSX_LETH16_V(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 5)
#define VSX_LETH32_V(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 6)
#define VSX_LETH64_V(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 7)
#define VSX_SETH8_V(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 4)
#define VSX_SETH16_V(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 5)
#define VSX_SETH32_V(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 6)
#define VSX_SETH64_V(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 7)

#define VSX_LGE8_V(VD, RS1, VS2) VSX_I_TYPE(VD, (RS1 & 0x1F), VS2, 8)
#define VSX_LGE16_V(VD, RS1, VS2) VSX_I_TYPE(VD, (RS1 & 0x1F), VS2, 9)
#define VSX_LGE32_V(VD, RS1, VS2) VSX_I_TYPE(VD, (RS1 & 0x1F), VS2, 10)
#define VSX_LGE64_V(VD, RS1, VS2) VSX_I_TYPE(VD, (RS1 & 0x1F), VS2, 11)
#define VSX_SSE8_V(VD, RS1, VS2) VSX_I_TYPE(VD, ((RS1 & 0x1F) | 0x20), VS2, 8)
#define VSX_SSE16_V(VD, RS1, VS2) VSX_I_TYPE(VD, ((RS1 & 0x1F) | 0x20), VS2, 9)
#define VSX_SSE32_V(VD, RS1, VS2) VSX_I_TYPE(VD, ((RS1 & 0x1F) | 0x20), VS2, 10)
#define VSX_SSE64_V(VD, RS1, VS2) VSX_I_TYPE(VD, ((RS1 & 0x1F) | 0x20), VS2, 11)

#define VSX_LGETH8_V(VD, RS1, VS2) VSX_I_TYPE(VD, (RS1 & 0x1F), VS2, 12)
#define VSX_LGETH16_V(VD, RS1, VS2) VSX_I_TYPE(VD, (RS1 & 0x1F), VS2, 13)
#define VSX_LGETH32_V(VD, RS1, VS2) VSX_I_TYPE(VD, (RS1 & 0x1F), VS2, 14)
#define VSX_LGETH64_V(VD, RS1, VS2) VSX_I_TYPE(VD, (RS1 & 0x1F), VS2, 15)
#define VSX_SSETH8_V(VD, RS1, VS2)                                             \
    VSX_I_TYPE(VD, ((RS1 & 0x1F) | 0x20), VS2, 12)
#define VSX_SSETH16_V(VD, RS1, VS2)                                            \
    VSX_I_TYPE(VD, ((RS1 & 0x1F) | 0x20), VS2, 13)
#define VSX_SSETH32_V(VD, RS1, VS2)                                            \
    VSX_I_TYPE(VD, ((RS1 & 0x1F) | 0x20), VS2, 14)
#define VSX_SSETH64_V(VD, RS1, VS2)                                            \
    VSX_I_TYPE(VD, ((RS1 & 0x1F) | 0x20), VS2, 15)

#define VSX_LE8_X(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 16)
#define VSX_LE16_X(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 17)
#define VSX_LE32_X(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 18)
#define VSX_LE64_X(VD, RS, IMM) VSX_I_TYPE(VD, (RS & 0x1F), IMM, 19)
#define VSX_SE8_X(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 16)
#define VSX_SE16_X(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 17)
#define VSX_SE32_X(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 18)
#define VSX_SE64_X(VD, RS, IMM) VSX_I_TYPE(VD, ((RS & 0x1F) | 0x20), IMM, 19)

#define VSX_ADD_VV(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 0, 20)
#define VSX_SUB_VV(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 1, 20)
#define VSX_MUL_VV(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 2, 20)
#define VSX_DIV_VV(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 3, 20)

#define VSX_ADD_VX(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 0, 21)
#define VSX_SUB_VX(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 1, 21)
#define VSX_MUL_VX(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 2, 21)
#define VSX_DIV_VX(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 3, 21)

#define VSX_AND_VV(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 0, 22)
#define VSX_OR_VV(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 1, 22)
#define VSX_XOR_VV(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 2, 22)

#define VSX_AND_VX(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 0, 23)
#define VSX_OR_VX(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 1, 23)
#define VSX_XOR_VX(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 2, 23)

#define VSX_MULA_VV(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 3, 22)
#define VSX_MULA_VX(VD, VS1, VS2) VSX_R_TYPE(VD, VS1, VS2, 3, 23)

#define VSX_TMUL(AD, VS1, VS2) VSX_R_TYPE((AD & 0x1F), VS1, VS2, 0, 24)
#define VSX_TMULA(AD, VS1, VS2)                                                \
    VSX_R_TYPE(((AD & 0x1F) | 0x20), VS1, VS2, 0, 24)

#define VSX_LT(AD, RS1, RS2)                                                   \
    VSX_R_TYPE((AD & 0x1F), (RS1 & 0x1F), (RS2 & 0x1F), 1, 24)
#define VSX_ST(AD, RS1, RS2)                                                   \
    VSX_R_TYPE(((AD & 0x1F) | 0x20), (RS1 & 0x1F), (RS2 & 0x1F), 1, 24)

#define VSX_TBRD(AD, VS) VSX_R_TYPE((AD & 0x1F), VS, 0, 2, 24)

#define VSX_SETVSX(RD, RS, IMM, IS_FLOAT)                                      \
    VSX_R_TYPE((RD & 0x1F), (RS & 0x1F), (IMM | IS_FLOAT), 0, 25)
#define VSX_MOV_VX(VD, RS) VSX_R_TYPE(VD, RS, 0, 0, 25)
#define VSX_MOV_VV(VD, VS) VSX_R_TYPE(VD, VS, 0, 1, 25)
