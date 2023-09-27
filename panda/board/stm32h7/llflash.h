bool flash_is_locked(void) {
  return (FLASH->CR1 & FLASH_CR_LOCK);
}

void flash_unlock(void) {
  FLASH->KEYR1 = 0x45670123;
  FLASH->KEYR1 = 0xCDEF89AB;
}

void flash_lock(void) {
  FLASH->CR1 |= FLASH_CR_LOCK;
}

bool flash_erase_sector(uint16_t sector) {
  #ifdef BOOTSTUB
    // don't erase the bootloader(sector 0)
    uint16_t min_sector = 1U;
    uint16_t max_sector = 7U;
  #else
    uint16_t min_sector = LOGGING_FLASH_SECTOR_A;
    uint16_t max_sector = LOGGING_FLASH_SECTOR_B;
  #endif

  bool ret = false;
  if ((sector >= min_sector) && (sector <= max_sector) && (!flash_is_locked())) {
    FLASH->CR1 = (sector << 8) | FLASH_CR_SER;
    FLASH->CR1 |= FLASH_CR_START;
    while ((FLASH->SR1 & FLASH_SR_QW) != 0U);
    ret = true;
  }
  return ret;
}

void flash_write_word(uint32_t *prog_ptr, uint32_t data) {
  #ifndef BOOTSTUB
  // don't write to any region besides the logging region
  if ((prog_ptr >= (uint32_t *)LOGGING_FLASH_BASE_A) && (prog_ptr < (uint32_t *)(LOGGING_FLASH_BASE_B + LOGGING_FLASH_SECTOR_SIZE))) {
  #endif

    uint32_t *pp = prog_ptr;
    FLASH->CR1 |= FLASH_CR_PG;
    *pp = data;
    while ((FLASH->SR1 & FLASH_SR_QW) != 0U);

  #ifndef BOOTSTUB
  }
  #endif
}

void flush_write_buffer(void) {
  if ((FLASH->SR1 & FLASH_SR_WBNE) != 0U) {
    FLASH->CR1 |= FLASH_CR_FW;
    while ((FLASH->SR1 & FLASH_CR_FW) != 0U);
  }
}
