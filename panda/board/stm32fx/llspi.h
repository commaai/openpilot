void llspi_miso_dma(uint8_t *addr, int len) {
  // disable DMA
  DMA2_Stream3->CR &= ~DMA_SxCR_EN;
  register_clear_bits(&(SPI1->CR2), SPI_CR2_TXDMAEN);

  // setup source and length
  register_set(&(DMA2_Stream3->M0AR), (uint32_t)addr, 0xFFFFFFFFU);
  DMA2_Stream3->NDTR = len;

  // enable DMA
  register_set_bits(&(SPI1->CR2), SPI_CR2_TXDMAEN);
  DMA2_Stream3->CR |= DMA_SxCR_EN;
}

void llspi_mosi_dma(uint8_t *addr, int len) {
  // disable DMA
  register_clear_bits(&(SPI1->CR2), SPI_CR2_RXDMAEN);
  DMA2_Stream2->CR &= ~DMA_SxCR_EN;

  // drain the bus
  volatile uint8_t dat = SPI1->DR;
  (void)dat;

  // setup destination and length
  register_set(&(DMA2_Stream2->M0AR), (uint32_t)addr, 0xFFFFFFFFU);
  DMA2_Stream2->NDTR = len;

  // enable DMA
  DMA2_Stream2->CR |= DMA_SxCR_EN;
  register_set_bits(&(SPI1->CR2), SPI_CR2_RXDMAEN);
}

// SPI MOSI DMA FINISHED
void DMA2_Stream2_IRQ_Handler(void) {
  // Clear interrupt flag
  ENTER_CRITICAL();
  DMA2->LIFCR = DMA_LIFCR_CTCIF2;

  spi_rx_done();

  EXIT_CRITICAL();
}

// SPI MISO DMA FINISHED
void DMA2_Stream3_IRQ_Handler(void) {
  // Clear interrupt flag
  DMA2->LIFCR = DMA_LIFCR_CTCIF3;

  // Wait until the transaction is actually finished and clear the DR
  // Timeout to prevent hang when the master clock stops.
  bool timed_out = false;
  uint32_t start_time = microsecond_timer_get();
  while (!(SPI1->SR & SPI_SR_TXE)) {
    if (get_ts_elapsed(microsecond_timer_get(), start_time) > SPI_TIMEOUT_US) {
      timed_out = true;
      break;
    }
  }
  volatile uint8_t dat = SPI1->DR;
  (void)dat;
  SPI1->DR = 0U;

  if (timed_out) {
    print("SPI: TX timeout\n");
  }

  spi_tx_done(timed_out);
}

// ***************************** SPI init *****************************
void llspi_init(void) {
  REGISTER_INTERRUPT(DMA2_Stream2_IRQn, DMA2_Stream2_IRQ_Handler, SPI_IRQ_RATE, FAULT_INTERRUPT_RATE_SPI_DMA)
  REGISTER_INTERRUPT(DMA2_Stream3_IRQn, DMA2_Stream3_IRQ_Handler, SPI_IRQ_RATE, FAULT_INTERRUPT_RATE_SPI_DMA)

  // Setup MOSI DMA
  register_set(&(DMA2_Stream2->CR), (DMA_SxCR_CHSEL_1 | DMA_SxCR_CHSEL_0 | DMA_SxCR_MINC | DMA_SxCR_TCIE), 0x1E077EFEU);
  register_set(&(DMA2_Stream2->PAR), (uint32_t)&(SPI1->DR), 0xFFFFFFFFU);

  // Setup MISO DMA
  register_set(&(DMA2_Stream3->CR), (DMA_SxCR_CHSEL_1 | DMA_SxCR_CHSEL_0 | DMA_SxCR_MINC | DMA_SxCR_DIR_0 | DMA_SxCR_TCIE), 0x1E077EFEU);
  register_set(&(DMA2_Stream3->PAR), (uint32_t)&(SPI1->DR), 0xFFFFFFFFU);

  // Enable SPI and the error interrupts
  // TODO: verify clock phase and polarity
  register_set(&(SPI1->CR1), SPI_CR1_SPE, 0xFFFFU);
  register_set(&(SPI1->CR2), 0U, 0xF7U);

  NVIC_EnableIRQ(DMA2_Stream2_IRQn);
  NVIC_EnableIRQ(DMA2_Stream3_IRQn);
}
