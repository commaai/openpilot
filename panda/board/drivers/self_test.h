#pragma once

#include "board/crc.h"

// UM2331 CPU_SM_0, RAM_SM_0/7/8, FLASH_SM_0/7/9
// at boot: all of RAM in board/stm32h7/self_test.s before the C runtime, then CPU, FPU and the whole flash image
// at 8Hz: CPU, FPU, one RAM block, an ECC scrub of the DMA RAMs and one flash chunk, ECC flags of every RAM in use
// UM2331 DMA_SM_3/4 at 8Hz: one DMA copy across the D1 to D2 interconnect, transfer errors of the sound DMAs, SPI's in llspi_queue

// keep in sync with board/stm32h7/self_test.s
#define SELF_TEST_PASS 0x5AFEC0DEU

// all IRQs share one priority, so each tick slice stays short
#define RAM_TEST_BLOCK_WORDS 64U
// consecutive blocks overlap by two words, for coupling across a block boundary
#define RAM_TEST_STEP ((RAM_TEST_BLOCK_WORDS - 2U) * 4U)
// kept clear of the frames of the running test
#define RAM_TEST_STACK_WINDOW 512U
// one alias per address bit above the block, AXI SRAM needs the most, 0x100 to 0x40000
#define RAM_TEST_ALIASES 11U
// SRAM1/2 and SRAM4 words read per tick for ECC scrubbing
#define RAM_SCRUB_WORDS 64U
#define FLASH_CRC_CHUNK 0x400U

extern uint32_t _dtcmram_start;
extern uint32_t _dtcmram_end;
extern uint32_t _axisram_start;
extern uint32_t _axisram_end;
extern uint32_t _itcmram_start;
extern uint32_t _itcmram_end;
extern uint32_t _sram12_start;
extern uint32_t _sram12_end;
extern uint32_t _sram4_start;
extern uint32_t _sram4_end;
extern uint32_t _stack_guard_end;
extern uint32_t self_test_cookie;

typedef struct {
  uint32_t start;
  uint32_t end;
} ram_region;

// RM0468 Table 11, every RAM in use, not the ETM and backup RAMs
static RAMECC_MonitorTypeDef *const ecc_monitors[] = {
  RAMECC1_Monitor1, RAMECC1_Monitor2, RAMECC1_Monitor3, RAMECC1_Monitor4, RAMECC1_Monitor6,
  RAMECC2_Monitor1, RAMECC2_Monitor2, RAMECC2_Monitor3, RAMECC3_Monitor1,
};
#define ECC_MONITOR_CNT (sizeof(ecc_monitors) / sizeof(ecc_monitors[0]))

#ifdef ALLOW_DEBUG
#define SELF_TEST_CPU 0U
#define SELF_TEST_RAM 1U
#define SELF_TEST_FLASH 2U
#define SELF_TEST_RAM_ECC 3U
#define SELF_TEST_FLASH_ECC 4U
#define SELF_TEST_DMA 5U
static uint8_t self_test_fail = 0U;
static uint32_t flash_crc_flip = 0U;

// the flash CRC injection goes through the CRC compare
void self_test_inject(uint8_t test) {
  if (test == SELF_TEST_FLASH) {
    flash_crc_flip = 1U;
  } else if (test <= SELF_TEST_DMA) {
    self_test_fail |= (uint8_t)(1U << test);
  } else {
  }
}

static bool self_test_injected(uint8_t test) {
  bool ret = (self_test_fail & (uint8_t)(1U << test)) != 0U;
  self_test_fail &= (uint8_t)~(1U << test);
  return ret;
}
#endif

// board/stm32h7/self_test.s
uint32_t cpu_test(void);
uint32_t fpu_test(void);

// march c-, word wide
static bool ram_test_march(volatile uint32_t *p, uint32_t n, uint32_t bg) {
  const uint32_t inv = ~bg;
  bool ok = true;
  for (uint32_t i = 0U; i < n; i++) {
    p[i] = bg;
  }
  for (uint32_t i = 0U; i < n; i++) {
    ok = ok && (p[i] == bg);
    p[i] = inv;
  }
  for (uint32_t i = 0U; i < n; i++) {
    ok = ok && (p[i] == inv);
    p[i] = bg;
  }
  for (uint32_t i = n; i > 0U; i--) {
    ok = ok && (p[i - 1U] == bg);
    p[i - 1U] = inv;
  }
  for (uint32_t i = n; i > 0U; i--) {
    ok = ok && (p[i - 1U] == inv);
    p[i - 1U] = bg;
  }
  for (uint32_t i = 0U; i < n; i++) {
    ok = ok && (p[i] == bg);
  }
  return ok;
}

// the stack guard faults on access and the running frames can't hold test patterns, both are covered at boot
static bool ram_test_skip(uint32_t start, uint32_t end) {
  uint32_t sp = __get_MSP();
  bool guard_hit = (start < (uint32_t)&_stack_guard_end) && (end > (uint32_t)&_stack_guard);
  bool stack_hit = (start < (sp + RAM_TEST_STACK_WINDOW)) && (end > (sp - RAM_TEST_STACK_WINDOW));
  return guard_hit || stack_hit;
}

// a decoder fault on an address bit above the block makes the block share cells with an alias, which then reads the patterns
static bool ram_test_aliases_ok(volatile uint32_t *const *alias, const uint32_t *val, uint8_t n) {
  bool ok = true;
  for (uint8_t i = 0U; i < n; i++) {
    ok = ok && (*alias[i] == val[i]);
  }
  return ok;
}

// transparent, interrupts stay off while the block holds test patterns
// the march covers the address decoder within the block, the aliases the address bits above it (UM2331 RAM_SM_8),
// checked after both backgrounds since an alias can hold one of them already
static bool ram_test_block(uint32_t addr, uint32_t words, uint32_t start, uint32_t end) {
  // outside the tested regions, copies are read back instead
  __attribute__((section(".sram12"))) static volatile uint32_t ram_test_backup[RAM_TEST_BLOCK_WORDS];
  volatile uint32_t *p = (uint32_t *)addr;
  volatile uint32_t *alias[RAM_TEST_ALIASES];
  uint32_t alias_val[RAM_TEST_ALIASES];
  uint8_t aliases = 0U;
  for (uint32_t bit = RAM_TEST_BLOCK_WORDS * 4U; (bit < (end - start)) && (aliases < RAM_TEST_ALIASES); bit <<= 1U) {
    uint32_t a = addr ^ bit;
    if ((a >= start) && ((a + 4U) <= end) && !ram_test_skip(a, a + 4U)) {
      alias[aliases] = (uint32_t *)a;
      aliases++;
    }
  }
  bool ok = true;
  ENTER_CRITICAL()
  // the block can hold the MPU protected safety state
  mpu_safety_open();
  for (uint32_t i = 0U; i < words; i++) {
    ram_test_backup[i] = p[i];
  }
  for (uint32_t i = 0U; i < words; i++) {
    ok = ok && (ram_test_backup[i] == p[i]);
  }
  for (uint8_t i = 0U; i < aliases; i++) {
    alias_val[i] = *alias[i];
  }
  ok = ram_test_march(p, words, 0x00000000U) && ok;
  ok = ram_test_aliases_ok(alias, alias_val, aliases) && ok;
  ok = ram_test_march(p, words, 0x55555555U) && ok;
  ok = ram_test_aliases_ok(alias, alias_val, aliases) && ok;
  for (uint32_t i = 0U; i < words; i++) {
    p[i] = ram_test_backup[i];
  }
  for (uint32_t i = 0U; i < words; i++) {
    ok = ok && (p[i] == ram_test_backup[i]);
  }
  mpu_safety_close();
  EXIT_CRITICAL()
  return ok;
}

// next block of the running pass, wraps around all regions, failed is set to the block address on a failure
static bool ram_test_next(uint32_t *failed) {
  // SRAM1/2 and SRAM4 hold the DMA buffers, which DMA can write mid test, and the test's own backup, they're covered at boot, by ECC and the scrub
  static const ram_region ram_regions[] = {
    {(uint32_t)&_dtcmram_start, (uint32_t)&_dtcmram_end},
    {(uint32_t)&_axisram_start, (uint32_t)&_axisram_end},
    {(uint32_t)&_itcmram_start, (uint32_t)&_itcmram_end},
  };
  static uint8_t region = 0U;
  static uint32_t block = 0U;
  if (block == 0U) {
    block = ram_regions[region].start;
  }
  uint32_t words = MIN(RAM_TEST_BLOCK_WORDS, (ram_regions[region].end - block) / 4U);
  bool ok = ram_test_skip(block, block + (words * 4U)) || ram_test_block(block, words, ram_regions[region].start, ram_regions[region].end);
  *failed = block;
  block += RAM_TEST_STEP;
  if (block >= ram_regions[region].end) {
    region = (uint8_t)((region + 1U) % (sizeof(ram_regions) / sizeof(ram_regions[0])));
    block = 0U;
  }
  return ok;
}

// UM2331 RAM_SM_7, a read reveals an ECC error in RAM that isn't march tested at runtime
static void ram_scrub_next(void) {
  static const ram_region scrub_regions[] = {
    {(uint32_t)&_sram12_start, (uint32_t)&_sram12_end},
    {(uint32_t)&_sram4_start, (uint32_t)&_sram4_end},
  };
  static uint8_t region = 0U;
  static uint32_t addr = 0U;
  if (addr == 0U) {
    addr = scrub_regions[region].start;
  }
  const volatile uint32_t *p = (const uint32_t *)addr;
  uint32_t words = MIN(RAM_SCRUB_WORDS, (scrub_regions[region].end - addr) / 4U);
  for (uint32_t i = 0U; i < words; i++) {
    (void)p[i];
  }
  addr += words * 4U;
  if (addr >= scrub_regions[region].end) {
    region = (uint8_t)((region + 1U) % (sizeof(scrub_regions) / sizeof(scrub_regions[0])));
    addr = 0U;
  }
}

// board/crypto/sign.py: length, image, crc32 of everything before it, version tag and version, signature
static uint32_t flash_crc_len(void) {
  uint32_t len = (uint32_t)_app_start[0];
  return ((len >= 12U) && (len < sizeof(_app_start))) ? (len - 12U) : 0U;
}

static uint32_t flash_crc_stored(void) {
  const uint8_t *p = &((const uint8_t *)_app_start)[flash_crc_len()];
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) | ((uint32_t)p[2] << 16U) | ((uint32_t)p[3] << 24U);
}

// crc is set to the computed CRC once a pass completes
static bool flash_test_next(uint32_t *crc_out) {
  static uint32_t offset = 0U;
  static uint32_t crc = 0xFFFFFFFFU;
  bool ok = true;
  uint32_t len = flash_crc_len();
  uint32_t n = MIN(FLASH_CRC_CHUNK, len - offset);
  crc = crc32_update(crc, &((const uint8_t *)_app_start)[offset], n);
  offset += n;
  if (offset >= len) {
#ifdef ALLOW_DEBUG
    crc ^= flash_crc_flip;
    flash_crc_flip = 0U;
#endif
    *crc_out = crc ^ 0xFFFFFFFFU;
    ok = (len != 0U) && (*crc_out == flash_crc_stored());
    offset = 0U;
    crc = 0xFFFFFFFFU;
  }
  return ok;
}

#define DMA_TEST_WORDS 8U
static const uint32_t dma_test_pattern[DMA_TEST_WORDS] = {
  0x5AA5F00FU, 0xA55A0FF0U, 0x12345678U, 0x9ABCDEF0U, 0x0F1E2D3CU, 0x4B5A6978U, 0x87968574U, 0xC3D2E1F0U,
};
__attribute__((section(".sram12"))) static uint32_t dma_test_buf[DMA_TEST_WORDS];

static void dma_test_start(void) {
  uint32_t len = DMA_TEST_WORDS;
#ifdef ALLOW_DEBUG
  // a short copy leaves the last word cleared
  if (self_test_injected(SELF_TEST_DMA)) {
    len -= 1U;
  }
#endif
  for (uint8_t i = 0U; i < DMA_TEST_WORDS; i++) {
    dma_test_buf[i] = 0U;
  }
  DMA2->HIFCR = DMA_HIFCR_CTCIF7 | DMA_HIFCR_CHTIF7 | DMA_HIFCR_CTEIF7 | DMA_HIFCR_CDMEIF7 | DMA_HIFCR_CFEIF7;
  DMA2_Stream7->NDTR = len;
  // the cleared buffer lands in SRAM1 before the DMA starts
  __DMB();
  DMA2_Stream7->CR |= DMA_SxCR_EN;
}

// flash to SRAM1, memory to memory needs the FIFO
static void dma_test_init(void) {
  register_set(&(DMA2_Stream7->PAR), (uint32_t)dma_test_pattern, 0xFFFFFFFFU);
  register_set(&(DMA2_Stream7->M0AR), (uint32_t)dma_test_buf, 0xFFFFFFFFU);
  register_set(&(DMA2_Stream7->FCR), DMA_SxFCR_DMDIS | DMA_SxFCR_FTH, DMA_SxFCR_DMDIS | DMA_SxFCR_FTH);
  register_set(&(DMA2_Stream7->CR), DMA_SxCR_DIR_1 | DMA_SxCR_PINC | DMA_SxCR_MINC | DMA_SxCR_PSIZE_1 | DMA_SxCR_MSIZE_1, 0x01FFFFFEU);
  dma_test_start();
}

// 0 if it passed, else bits 0-7 the wrong words, 8-11 DMA1 stream 0/1 and BDMA channel 0/1 errors, 22-27 the DMA2 stream 7 flags with TCIF7 if not done
static uint32_t dma_test_next(void) {
  uint32_t err = (DMA2->HISR & (DMA_HISR_TCIF7 | DMA_HISR_TEIF7 | DMA_HISR_DMEIF7 | DMA_HISR_FEIF7)) ^ DMA_HISR_TCIF7;
  for (uint8_t i = 0U; i < DMA_TEST_WORDS; i++) {
    err |= (dma_test_buf[i] != dma_test_pattern[i]) ? (1UL << i) : 0U;
  }
  // mic on DMA1 stream 0, sound and siren on DMA1 stream 1, SAI4 on BDMA channels 0 and 1, a sound stop clears them in between
  err |= ((DMA1->LISR & DMA_LISR_TEIF0) != 0U) ? (1UL << 8) : 0U;
  err |= ((DMA1->LISR & DMA_LISR_TEIF1) != 0U) ? (1UL << 9) : 0U;
  err |= ((BDMA->ISR & BDMA_ISR_TEIF0) != 0U) ? (1UL << 10) : 0U;
  err |= ((BDMA->ISR & BDMA_ISR_TEIF1) != 0U) ? (1UL << 11) : 0U;
  dma_test_start();
  return err;
}

// ram_addr is the failing block, 0 at boot, flash_crc the computed CRC
static void self_test_check(bool cpu_ok, bool ram_ok, uint32_t ram_addr, bool flash_ok, uint32_t flash_crc) {
  bool cpu = cpu_ok;
  bool ram = ram_ok;
  bool flash = flash_ok;
#ifdef ALLOW_DEBUG
  cpu = cpu && !self_test_injected(SELF_TEST_CPU);
  ram = ram && !self_test_injected(SELF_TEST_RAM);
#endif
  if (!cpu) {
    fault_occurred(FAULT_SELF_TEST_CPU);
  }
  if (!ram) {
    fault_occurred_value(FAULT_SELF_TEST_RAM, ram_addr);
  }
  if (!flash) {
    fault_occurred_value(FAULT_SELF_TEST_FLASH, flash_crc);
  }
  uint32_t flash_sr = FLASH->SR1;
  uint32_t ecc_sr = 0U;
  // the monitor index in the top 4 bits and the failing word within that RAM
  uint32_t ecc_far = 0U;
  for (uint8_t i = 0U; i < ECC_MONITOR_CNT; i++) {
    uint32_t sr = ecc_monitors[i]->SR;
    if ((ecc_far == 0U) && ((sr & (RAMECC_SR_DEDF | RAMECC_SR_DEBWDF)) != 0U)) {
      ecc_far = ((uint32_t)i << 28U) | (ecc_monitors[i]->FAR & 0x0FFFFFFFU);
    }
    ecc_sr |= sr;
  }
#ifdef ALLOW_DEBUG
  if (self_test_injected(SELF_TEST_FLASH_ECC)) {
    flash_sr |= FLASH_SR_DBECCERR;
  }
  if (self_test_injected(SELF_TEST_RAM_ECC)) {
    ecc_sr |= RAMECC_SR_DEDF;
  }
#endif
  // a CPU read with a double ECC error bus faults, this catches the DMA reads from flash
  if ((flash_sr & FLASH_SR_DBECCERR) != 0U) {
    fault_occurred_value(FAULT_ECC, FLASH->ECC_FA1);
  }
  if ((ecc_sr & (RAMECC_SR_DEDF | RAMECC_SR_DEBWDF)) != 0U) {
    fault_occurred_value(FAULT_ECC, ecc_far);
  }
}

void self_test_init(void) {
  crc32_init();

  // the boot RAM test wrote every word, so no RAM holds an unset ECC code anymore
  for (uint8_t i = 0U; i < ECC_MONITOR_CNT; i++) {
    ecc_monitors[i]->SR = 0U;
    register_set(&(ecc_monitors[i]->CR), RAMECC_CR_ECCELEN, 0x3CU);
  }

  bool ram_ok = (self_test_cookie == SELF_TEST_PASS);
  self_test_cookie = 0U;
  uint32_t len = flash_crc_len();
  uint32_t crc = crc32((const uint8_t *)_app_start, len);
  bool flash_ok = (len != 0U) && (crc == flash_crc_stored());
  self_test_check((cpu_test() == SELF_TEST_PASS) && (fpu_test() == SELF_TEST_PASS), ram_ok, 0U, flash_ok, crc);
  dma_test_init();
}

// called at 8Hz
void self_test_tick(void) {
  bool cpu_ok = (cpu_test() == SELF_TEST_PASS) && (fpu_test() == SELF_TEST_PASS);
  uint32_t ram_addr = 0U;
  bool ram_ok = ram_test_next(&ram_addr);
  ram_scrub_next();
  uint32_t crc = 0U;
  bool flash_ok = flash_test_next(&crc);
  self_test_check(cpu_ok, ram_ok, ram_addr, flash_ok, crc);
  uint32_t dma_err = dma_test_next();
  if (dma_err != 0U) {
    fault_occurred_value(FAULT_DMA, dma_err);
  }
}
