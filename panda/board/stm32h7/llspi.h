// master -> panda DMA start
void llspi_mosi_dma(uint8_t *addr, int len) {
  // disable DMA + SPI
  register_clear_bits(&(SPI4->CFG1), SPI_CFG1_RXDMAEN);
  DMA2_Stream2->CR &= ~DMA_SxCR_EN;
  register_clear_bits(&(SPI4->CR1), SPI_CR1_SPE);

  // drain the bus
  while ((SPI4->SR & SPI_SR_RXP) != 0U) {
    volatile uint8_t dat = SPI4->RXDR;
    (void)dat;
  }

  // setup destination and length
  register_set(&(DMA2_Stream2->M0AR), (uint32_t)addr, 0xFFFFFFFFU);
  DMA2_Stream2->NDTR = len;

  // enable DMA + SPI
  DMA2_Stream2->CR |= DMA_SxCR_EN;
  register_set_bits(&(SPI4->CFG1), SPI_CFG1_RXDMAEN);
  register_set_bits(&(SPI4->CR1), SPI_CR1_SPE);
}

// panda -> master DMA start
void llspi_miso_dma(uint8_t *addr, int len) {
  // disable DMA
  DMA2_Stream3->CR &= ~DMA_SxCR_EN;
  register_clear_bits(&(SPI4->CFG1), SPI_CFG1_TXDMAEN);

  // setup source and length
  register_set(&(DMA2_Stream3->M0AR), (uint32_t)addr, 0xFFFFFFFFU);
  DMA2_Stream3->NDTR = len;

  // clear under-run while we were reading
  SPI4->IFCR |= SPI_IFCR_UDRC;

  // enable DMA
  register_set_bits(&(SPI4->CFG1), SPI_CFG1_TXDMAEN);
  DMA2_Stream3->CR |= DMA_SxCR_EN;
}

// master -> panda DMA finished
void DMA2_Stream2_IRQ_Handler(void) {
  // Clear interrupt flag
  ENTER_CRITICAL();
  DMA2->LIFCR = DMA_LIFCR_CTCIF2;

  spi_handle_rx();

  EXIT_CRITICAL();
}

// panda -> master DMA finished
void DMA2_Stream3_IRQ_Handler(void) {
  // Clear interrupt flag
  DMA2->LIFCR = DMA_LIFCR_CTCIF3;

  // Wait until the transaction is actually finished and clear the DR.
  // Timeout to prevent hang when the master clock stops.
  bool timed_out = false;
  uint32_t start_time = microsecond_timer_get();  
  while (!(SPI4->SR & SPI_SR_TXC)) {
    if (get_ts_elapsed(microsecond_timer_get(), start_time) > SPI_TIMEOUT_US) {
      timed_out = true;
      break;
    }
  }
  volatile uint8_t dat = SPI4->TXDR;
  (void)dat;

  spi_handle_tx(timed_out);
}


void llspi_init(void) {
  // We expect less than 50 transactions (including control messages and CAN buffers) at the 100Hz boardd interval. Can be raised if needed.
  REGISTER_INTERRUPT(DMA2_Stream2_IRQn, DMA2_Stream2_IRQ_Handler, 5000U, FAULT_INTERRUPT_RATE_SPI_DMA)
  REGISTER_INTERRUPT(DMA2_Stream3_IRQn, DMA2_Stream3_IRQ_Handler, 5000U, FAULT_INTERRUPT_RATE_SPI_DMA)

  // Setup MOSI DMA
  register_set(&(DMAMUX1_Channel10->CCR), 83U, 0xFFFFFFFFU);
  register_set(&(DMA2_Stream2->CR), (DMA_SxCR_MINC | DMA_SxCR_TCIE), 0x1E077EFEU);
  register_set(&(DMA2_Stream2->PAR), (uint32_t)&(SPI4->RXDR), 0xFFFFFFFFU);

  // Setup MISO DMA, memory -> peripheral
  register_set(&(DMAMUX1_Channel11->CCR), 84U, 0xFFFFFFFFU);
  register_set(&(DMA2_Stream3->CR), (DMA_SxCR_MINC | DMA_SxCR_DIR_0 | DMA_SxCR_TCIE), 0x1E077EFEU);
  register_set(&(DMA2_Stream3->PAR), (uint32_t)&(SPI4->TXDR), 0xFFFFFFFFU);

  // Enable SPI
  register_set(&(SPI4->CFG1), (7U << SPI_CFG1_DSIZE_Pos), SPI_CFG1_DSIZE_Msk);
  register_set(&(SPI4->UDRDR), 0xcd, 0xFFFFU);  // set under-run value for debugging
  register_set(&(SPI4->CR1), SPI_CR1_SPE, 0xFFFFU);
  register_set(&(SPI4->CR2), 0, 0xFFFFU);

  NVIC_EnableIRQ(DMA2_Stream2_IRQn);
  NVIC_EnableIRQ(DMA2_Stream3_IRQn);
}
