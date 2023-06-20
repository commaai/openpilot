// minimal code to fake a panda for tests
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "utils.h"
#include "drivers/rtc_definitions.h"

#define CANFD
#define ALLOW_DEBUG
#define PANDA

#define ENTER_CRITICAL() 0
#define EXIT_CRITICAL() 0

void print(const char *a) {
  printf("%s", a);
}

void puth(unsigned int i) {
  printf("%u", i);
}

typedef struct {
  uint32_t CNT;
} TIM_TypeDef;

TIM_TypeDef timer;
TIM_TypeDef *MICROSECOND_TIMER = &timer;
uint32_t microsecond_timer_get(void);

uint32_t microsecond_timer_get(void) {
  return MICROSECOND_TIMER->CNT;
}

// Register functions
void register_set_bits(volatile uint32_t *addr, uint32_t val) {}

// RTC
timestamp_t rtc_get_time() {
  timestamp_t result;
  result.year = 1996;
  result.month = 4;
  result.day = 23;
  result.weekday = 2;
  result.hour = 4;
  result.minute = 20;
  result.second = 20;
  return result;
}

// Logging and flash
uint8_t fake_logging_bank[0x40000] __attribute__ ((aligned (4)));
#define LOGGING_FLASH_BASE_A (&fake_logging_bank[0])
#define LOGGING_FLASH_BASE_B (&fake_logging_bank[0x20000])
#define LOGGING_FLASH_SECTOR_A 5
#define LOGGING_FLASH_SECTOR_B 6
#define LOGGING_FLASH_SECTOR_SIZE 0x20000U

bool flash_locked = true;
void flash_unlock(void) {
  flash_locked = false;
}
void flash_lock(void) {
  flash_locked = true;
}

void *memset(void *str, int c, unsigned int n);

bool flash_erase_sector(uint8_t sector) {
  if (flash_locked) {
    return false;
  }

  switch (sector) {
    case LOGGING_FLASH_SECTOR_A:
      memset(LOGGING_FLASH_BASE_A, 0xFF, sizeof(fake_logging_bank)/2);
      return true;
    case LOGGING_FLASH_SECTOR_B:
      memset(LOGGING_FLASH_BASE_B, 0xFF, sizeof(fake_logging_bank)/2);
      return true;
    default:
      return false;
  }
}

void flash_write_word(void *prog_ptr, uint32_t data) {
  if (flash_locked || prog_ptr < (void *) LOGGING_FLASH_BASE_A || prog_ptr >= (void *) (LOGGING_FLASH_BASE_A + sizeof(fake_logging_bank))) {
    return;
  }

  *(uint32_t *)prog_ptr = data;
}

void flush_write_buffer(void) {}