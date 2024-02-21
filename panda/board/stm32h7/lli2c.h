
// TODO: this driver relies heavily on polling,
// if we want it to be more async, we should use interrupts

#define I2C_TIMEOUT_US 100000U

// cppcheck-suppress misra-c2012-2.7; not sure why it triggers here?
bool i2c_status_wait(const volatile uint32_t *reg, uint32_t mask, uint32_t val) {
  uint32_t start_time = microsecond_timer_get();
  while(((*reg & mask) != val) && (get_ts_elapsed(microsecond_timer_get(), start_time) < I2C_TIMEOUT_US));
  return ((*reg & mask) == val);
}

bool i2c_write_reg(I2C_TypeDef *I2C, uint8_t addr, uint8_t reg, uint8_t value) {
  // Setup transfer and send START + addr
  bool ret = false;
  for(uint32_t i=0U; i<10U; i++) {
    register_clear_bits(&I2C->CR2, I2C_CR2_ADD10);
    I2C->CR2 = ((addr << 1U) & I2C_CR2_SADD_Msk);
    register_clear_bits(&I2C->CR2, I2C_CR2_RD_WRN);
    register_set_bits(&I2C->CR2, I2C_CR2_AUTOEND);
    I2C->CR2 |= (2 << I2C_CR2_NBYTES_Pos);

    I2C->CR2 |= I2C_CR2_START;
    if(!i2c_status_wait(&I2C->CR2, I2C_CR2_START, 0U)) {
      continue;
    }

    // check if we lost arbitration
    if ((I2C->ISR & I2C_ISR_ARLO) != 0U) {
      register_set_bits(&I2C->ICR, I2C_ICR_ARLOCF);
    } else {
      ret = true;
      break;
    }
  }

  if (!ret) {
    goto end;
  }

  // Send data
  ret = i2c_status_wait(&I2C->ISR, I2C_ISR_TXIS, I2C_ISR_TXIS);
  if(!ret) {
    goto end;
  }
  I2C->TXDR = reg;

  ret = i2c_status_wait(&I2C->ISR, I2C_ISR_TXIS, I2C_ISR_TXIS);
  if(!ret) {
    goto end;
  }
  I2C->TXDR = value;

end:
  return ret;
}

bool i2c_read_reg(I2C_TypeDef *I2C, uint8_t addr, uint8_t reg, uint8_t *value) {
  // Setup transfer and send START + addr
  bool ret = false;
  for(uint32_t i=0U; i<10U; i++) {
    register_clear_bits(&I2C->CR2, I2C_CR2_ADD10);
    I2C->CR2 = ((addr << 1U) & I2C_CR2_SADD_Msk);
    register_clear_bits(&I2C->CR2, I2C_CR2_RD_WRN);
    register_clear_bits(&I2C->CR2, I2C_CR2_AUTOEND);
    I2C->CR2 |= (1 << I2C_CR2_NBYTES_Pos);

    I2C->CR2 |= I2C_CR2_START;
    if(!i2c_status_wait(&I2C->CR2, I2C_CR2_START, 0U)) {
      continue;
    }

    // check if we lost arbitration
    if ((I2C->ISR & I2C_ISR_ARLO) != 0U) {
      register_set_bits(&I2C->ICR, I2C_ICR_ARLOCF);
    } else {
      ret = true;
      break;
    }
  }

  if (!ret) {
    goto end;
  }

  // Send data
  ret = i2c_status_wait(&I2C->ISR, I2C_ISR_TXIS, I2C_ISR_TXIS);
  if(!ret) {
    goto end;
  }
  I2C->TXDR = reg;

  // Restart
  I2C->CR2 = (((addr << 1) | 0x1U) & I2C_CR2_SADD_Msk) | (1U << I2C_CR2_NBYTES_Pos) | I2C_CR2_RD_WRN | I2C_CR2_START;
  ret = i2c_status_wait(&I2C->CR2, I2C_CR2_START, 0U);
  if(!ret) {
    goto end;
  }

  // check if we lost arbitration
  if ((I2C->ISR & I2C_ISR_ARLO) != 0U) {
    register_set_bits(&I2C->ICR, I2C_ICR_ARLOCF);
    ret = false;
    goto end;
  }

  // Read data
  ret = i2c_status_wait(&I2C->ISR, I2C_ISR_RXNE, I2C_ISR_RXNE);
  if(!ret) {
    goto end;
  }
  *value = I2C->RXDR;

  // Stop
  I2C->CR2 |= I2C_CR2_STOP;

end:
  return ret;
}

bool i2c_set_reg_bits(I2C_TypeDef *I2C, uint8_t address, uint8_t regis, uint8_t bits) {
  uint8_t value;
  bool ret = i2c_read_reg(I2C, address, regis, &value);
  if(ret) {
    ret = i2c_write_reg(I2C, address, regis, value | bits);
  }
  return ret;
}

bool i2c_clear_reg_bits(I2C_TypeDef *I2C, uint8_t address, uint8_t regis, uint8_t bits) {
  uint8_t value;
  bool ret = i2c_read_reg(I2C, address, regis, &value);
  if(ret) {
    ret = i2c_write_reg(I2C, address, regis, value & (uint8_t) (~bits));
  }
  return ret;
}

bool i2c_set_reg_mask(I2C_TypeDef *I2C, uint8_t address, uint8_t regis, uint8_t value, uint8_t mask) {
  uint8_t old_value;
  bool ret = i2c_read_reg(I2C, address, regis, &old_value);
  if(ret) {
    ret = i2c_write_reg(I2C, address, regis, (old_value & (uint8_t) (~mask)) | (value & mask));
  }
  return ret;
}

void i2c_init(I2C_TypeDef *I2C) {
  // 100kHz clock speed
  I2C->TIMINGR = 0x107075B0;
  I2C->CR1 = I2C_CR1_PE;
}