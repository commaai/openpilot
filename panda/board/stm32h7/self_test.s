  .syntax unified
  .cpu cortex-m7
  .thumb

/* keep in sync with board/drivers/self_test.h */
  .equ SELF_TEST_PASS, 0x5AFEC0DE

/* r0 holds the expected value */
.macro expect reg, val, fail
  ldr r0, =\val
  cmp \reg, r0
  bne \fail
.endm

  .section .text.cpu_test
  .global cpu_test
  .type cpu_test, %function

/* UM2331 CPU_SM_0, walking bit (IEC 61508-7 A.3.2) and a fixed ALU, flags, branch and load/store sequence
   returns SELF_TEST_PASS, anything else is a fault */
cpu_test:
  push {r4-r11, lr}
  mrs r1, primask
  mrs r2, psp
  push {r1, r2}
  cpsid i

  /* every register its own value, written before any is read back, so coupling between registers shows up */
  .irp pattern, 0x55555555, 0xAAAAAAAA
  ldr r0, =\pattern
  .irp reg, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr
  add r0, r0, #0x01010101
  mov \reg, r0
  .endr
  ldr r0, =\pattern
  .irp reg, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr
  add r0, r0, #0x01010101
  cmp \reg, r0
  bne .Lcpu_fail
  .endr
  ldr r0, =\pattern
  ldr r1, =\pattern
  cmp r0, r1
  bne .Lcpu_fail
  .endr

  /* walking one, against r0 */
  .irp reg, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr
  movs r0, #1
  mov \reg, #1
1:
  cmp \reg, r0
  bne .Lcpu_fail
  lsl \reg, \reg, #1
  lsls r0, r0, #1
  bne 1b
  .endr

  /* stack pointers, MSP is word aligned and only out of use for the two instructions in between */
  .irp pattern, 0x55555554, 0xAAAAAAA8
  ldr r1, =\pattern
  msr psp, r1
  mrs r2, psp
  cmp r2, r1
  bne .Lcpu_fail
  mov r3, sp
  msr msp, r1
  mrs r2, msp
  msr msp, r3
  cmp r2, r1
  bne .Lcpu_fail
  .endr

  /* APSR NZCVQ and GE */
  .irp flags, 0xA80A0000, 0x50050000
  ldr r1, =\flags
  msr APSR_nzcvqg, r1
  mrs r2, APSR
  ldr r3, =0xF80F0000
  and r2, r2, r3
  cmp r2, r1
  bne .Lcpu_fail
  .endr

  /* ALU, expected values computed off target */
  ldr r4, =0x12345678
  ldr r5, =0x9ABCDEF0
  mul r1, r4, r5
  expect r1, 0x242D2080, .Lcpu_fail
  umull r1, r2, r4, r5
  expect r1, 0x242D2080, .Lcpu_fail
  expect r2, 0x0B00EA4E, .Lcpu_fail
  smull r1, r2, r4, r5
  expect r2, 0xF8CC93D6, .Lcpu_fail
  mla r1, r4, r5, r4
  expect r1, 0x366176F8, .Lcpu_fail
  udiv r1, r5, r4
  expect r1, 0x00000008, .Lcpu_fail
  sdiv r1, r5, r4
  expect r1, 0xFFFFFFFB, .Lcpu_fail
  ror r1, r4, #8
  expect r1, 0x78123456, .Lcpu_fail
  asr r1, r5, #4
  expect r1, 0xF9ABCDEF, .Lcpu_fail
  lsr r1, r5, #4
  expect r1, 0x09ABCDEF, .Lcpu_fail
  lsl r1, r4, #4
  expect r1, 0x23456780, .Lcpu_fail
  eor r1, r4, r5
  expect r1, 0x88888888, .Lcpu_fail
  and r1, r4, r5
  expect r1, 0x12345670, .Lcpu_fail
  orr r1, r4, r5
  expect r1, 0x9ABCDEF8, .Lcpu_fail
  bic r1, r4, r5
  expect r1, 0x00000008, .Lcpu_fail
  mvn r1, r4
  expect r1, 0xEDCBA987, .Lcpu_fail
  clz r1, r4
  expect r1, 0x00000003, .Lcpu_fail
  rbit r1, r4
  expect r1, 0x1E6A2C48, .Lcpu_fail
  rev r1, r4
  expect r1, 0x78563412, .Lcpu_fail

  /* flags and conditional branches */
  ldr r1, =0x7FFFFFFF
  adds r1, r1, #1
  bvc .Lcpu_fail
  bpl .Lcpu_fail
  bcs .Lcpu_fail
  beq .Lcpu_fail
  adds r1, r1, r1
  bne .Lcpu_fail
  bcc .Lcpu_fail
  bvc .Lcpu_fail
  bmi .Lcpu_fail
  movs r1, #0
  subs r1, r1, #1
  bcs .Lcpu_fail
  bpl .Lcpu_fail
  bvs .Lcpu_fail
  cmp r4, r5
  ite lo
  movlo r1, #1
  movhs r1, #2
  expect r1, 1, .Lcpu_fail
  cmp r4, r5
  ite gt
  movgt r1, #1
  movle r1, #2
  expect r1, 1, .Lcpu_fail

  /* load and store widths */
  sub sp, sp, #16
  str r4, [sp]
  strh r5, [sp, #4]
  strb r4, [sp, #6]
  strd r4, r5, [sp, #8]
  ldr r1, [sp]
  ldrh r2, [sp, #4]
  ldrb r3, [sp, #6]
  ldrd r6, r7, [sp, #8]
  add sp, sp, #16
  expect r1, 0x12345678, .Lcpu_fail
  expect r2, 0x0000DEF0, .Lcpu_fail
  expect r3, 0x00000078, .Lcpu_fail
  cmp r6, r4
  bne .Lcpu_fail
  cmp r7, r5
  bne .Lcpu_fail

  ldr r0, =SELF_TEST_PASS
  b .Lcpu_done
.Lcpu_fail:
  movs r0, #0
.Lcpu_done:
  pop {r1, r2}
  msr psp, r2
  msr primask, r1
  pop {r4-r11, pc}
  .ltorg
  .size cpu_test, .-cpu_test

  .fpu fpv5-d16
  .section .text.fpu_test
  .global fpu_test
  .type fpu_test, %function

/* UM2331 CPU_SM_0 for the FPU: every register its own value, FPSCR flags and control bits, arithmetic
   returns SELF_TEST_PASS, anything else is a fault */
fpu_test:
  vpush {s16-s31}
  vmrs r3, fpscr

  .irp pattern, 0x55555555, 0xAAAAAAAA
  ldr r0, =\pattern
  .irp reg, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31
  add r0, r0, #0x01010101
  vmov \reg, r0
  .endr
  ldr r0, =\pattern
  .irp reg, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31
  add r0, r0, #0x01010101
  vmov r1, \reg
  cmp r1, r0
  bne .Lfpu_fail
  .endr
  .endr

  /* NZCV and AHP, DN, FZ, RMode */
  .irp bits, 0xA6800000, 0x51400000
  ldr r1, =\bits
  vmsr fpscr, r1
  vmrs r2, fpscr
  ldr r0, =0xF7C00000
  and r2, r2, r0
  cmp r2, r1
  bne .Lfpu_fail
  .endr

  /* exact single precision results, round to nearest */
  vmsr fpscr, r3
  ldr r0, =0x3FC00000
  vmov s0, r0
  ldr r0, =0x40100000
  vmov s1, r0
  vadd.f32 s2, s0, s1
  vmov r1, s2
  expect r1, 0x40700000, .Lfpu_fail
  vsub.f32 s2, s1, s0
  vmov r1, s2
  expect r1, 0x3F400000, .Lfpu_fail
  vmul.f32 s2, s0, s1
  vmov r1, s2
  expect r1, 0x40580000, .Lfpu_fail
  vdiv.f32 s3, s2, s0
  vmov r1, s3
  expect r1, 0x40100000, .Lfpu_fail
  vsqrt.f32 s3, s1
  vmov r1, s3
  expect r1, 0x3FC00000, .Lfpu_fail
  vmov.f32 s3, s0
  vfma.f32 s3, s0, s1
  vmov r1, s3
  expect r1, 0x409C0000, .Lfpu_fail
  vcmp.f32 s0, s1
  vmrs APSR_nzcv, fpscr
  bge .Lfpu_fail

  ldr r0, =SELF_TEST_PASS
  b .Lfpu_done
.Lfpu_fail:
  movs r0, #0
.Lfpu_done:
  vmsr fpscr, r3
  vpop {s16-s31}
  bx lr
  .ltorg
  .size fpu_test, .-fpu_test

  .section .text.ram_test_startup
  .global ram_test_startup
  .type ram_test_startup, %function

/* UM2331 RAM_SM_0/8, march c- over all RAM before the C runtime, without a stack since the stack is tested too
   the first element writes 64 bits, which also sets up the ECC of AXI SRAM and ITCM (64-bit ECC words, RM0468 3.2)
   leaves the RAM zeroed and SELF_TEST_PASS in self_test_cookie if every word passed */
ram_test_startup:
  /* SRAM1/2 clocks are off after reset (RM0468 RCC_AHB2ENR), read back before the first access (RM0468 8.5.10) */
  ldr r0, =0x580244DC
  ldr r1, [r0]
  orr r1, r1, #0x60000000
  str r1, [r0]
  ldr r1, [r0]
  dsb

  /* the reset cookies are in SRAM4 */
  ldr r0, =enter_bootloader_mode
  ldr r10, [r0]
  ldr r0, =fault_cookie
  ldr r11, [r0]
  ldr r0, =fault_cfsr
  ldr r7, [r0]
  ldr r0, =fault_addr
  ldr r12, [r0]
  ldr r0, =fault_pc
  ldr r4, [r0]

  ldr r9, =SELF_TEST_PASS
  ldr r8, =ram_test_regions
.Lram_region:
  ldr r0, [r8], #4
  ldr r1, [r8], #4
  cmp r0, r1
  beq .Lram_done
  .irp bg, 0x55555555, 0x00000000
  ldr r2, =\bg
  mvn r3, r2
  mov r5, r0
1:
  strd r2, r2, [r5], #8
  cmp r5, r1
  blo 1b
  mov r5, r0
2:
  ldr r6, [r5]
  cmp r6, r2
  it ne
  movne r9, #0
  str r3, [r5], #4
  cmp r5, r1
  blo 2b
  mov r5, r0
3:
  ldr r6, [r5]
  cmp r6, r3
  it ne
  movne r9, #0
  str r2, [r5], #4
  cmp r5, r1
  blo 3b
  mov r5, r1
4:
  ldr r6, [r5, #-4]!
  cmp r6, r2
  it ne
  movne r9, #0
  str r3, [r5]
  cmp r5, r0
  bhi 4b
  mov r5, r1
5:
  ldr r6, [r5, #-4]!
  cmp r6, r3
  it ne
  movne r9, #0
  str r2, [r5]
  cmp r5, r0
  bhi 5b
  mov r5, r0
6:
  ldr r6, [r5], #4
  cmp r6, r2
  it ne
  movne r9, #0
  cmp r5, r1
  blo 6b
  .endr
  b .Lram_region

.Lram_done:
  ldr r0, =enter_bootloader_mode
  str r10, [r0]
  ldr r0, =fault_cookie
  str r11, [r0]
  ldr r0, =fault_cfsr
  str r7, [r0]
  ldr r0, =fault_addr
  str r12, [r0]
  ldr r0, =fault_pc
  str r4, [r0]
  ldr r0, =self_test_cookie
  str r9, [r0]
  bx lr
  .ltorg
  .size ram_test_startup, .-ram_test_startup

  .section .rodata.ram_test_regions
  .align 2
ram_test_regions:
  .word _dtcmram_start, _dtcmram_end
  .word _axisram_start, _axisram_end
  .word _itcmram_start, _itcmram_end
  .word _sram12_start, _sram12_end
  .word _sram4_start, _sram4_end
  .word 0, 0
