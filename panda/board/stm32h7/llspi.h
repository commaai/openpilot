#pragma once

// SPI4 NSS (chip select) is PE11
#define SPI_CS_PIN 11U

void spi_cs_rising(void);

// Queues the next transfer: MOSI into spi_buf_rx, MISO from tx. A transfer that's already in progress is ignored,
// or at worst garbled, which the protocol recovers from
void llspi_queue(const uint8_t *tx, uint16_t tx_len) {
  // stop DMA + SPI
  register_clear_bits(&(SPI4->CR1), SPI_CR1_SPE);
  DMA2_Stream2->CR &= ~DMA_SxCR_EN;
  DMA2_Stream3->CR &= ~DMA_SxCR_EN;
  register_clear_bits(&(SPI4->CFG1), SPI_CFG1_RXDMAEN | SPI_CFG1_TXDMAEN);
  for (uint16_t i = 0U; (i < 1000U) && (((DMA2_Stream2->CR & DMA_SxCR_EN) != 0U) || ((DMA2_Stream3->CR & DMA_SxCR_EN) != 0U)); i++) {}

  // drain the RX FIFO and clear all flags
  while ((SPI4->SR & SPI_SR_RXP) != 0U) {
    volatile uint8_t dat = SPI4->RXDR;
    (void)dat;
  }
  SPI4->IFCR = (0x1FFU << 3U);
  // DMA_SM_4: a transfer error of the last transfer, checked before the clear
  if ((DMA2->LISR & (DMA_LISR_TEIF2 | DMA_LISR_TEIF3)) != 0U) {
    fault_occurred_value(FAULT_DMA, DMA2->LISR);
  }
  DMA2->LIFCR = (0x3DUL << 16U) | (0x3DUL << 22U);  // streams 2 + 3

  // setup buffers
  register_set(&(DMA2_Stream2->M0AR), (uint32_t)spi_buf_rx, 0xFFFFFFFFU);
  DMA2_Stream2->NDTR = SPI_BUF_SIZE;
  register_set(&(DMA2_Stream3->M0AR), (uint32_t)tx, 0xFFFFFFFFU);
  DMA2_Stream3->NDTR = tx_len;

  // enable, in the order from the reference manual
  register_set_bits(&(SPI4->CFG1), SPI_CFG1_RXDMAEN);
  DMA2_Stream2->CR |= DMA_SxCR_EN;
  DMA2_Stream3->CR |= DMA_SxCR_EN;
  register_set_bits(&(SPI4->CFG1), SPI_CFG1_TXDMAEN);
  register_set_bits(&(SPI4->CR1), SPI_CR1_SPE);
}

// Number of bytes received since we last queued, saturates at SPI_BUF_SIZE
uint16_t llspi_rx_count(void) {
  // let the DMA empty the FIFO, unless it's already full
  for (uint16_t i = 0U; (i < 1000U) && ((SPI4->SR & SPI_SR_RXP) != 0U) && (DMA2_Stream2->NDTR != 0U); i++) {}
  return SPI_BUF_SIZE - DMA2_Stream2->NDTR;
}

#define EXTI15_10_LINES 0xFC00U

// CS rising edge, the end of a transfer
static void EXTI15_10_IRQ_Handler(void) {
  if ((EXTI->PR1 & (1UL << SPI_CS_PIN)) != 0U) {
    // clear first, so an edge during processing fires again, read back so the clear lands before the return
    EXTI->PR1 = (1UL << SPI_CS_PIN);
    (void)EXTI->PR1;
    // if we're late and the next transfer already started, its own rising edge handles it
    if ((GPIOE->IDR & (1UL << SPI_CS_PIN)) != 0U) {
      spi_cs_rising();
    }
  }
  // no other line on this vector has a trigger enabled
  if ((EXTI->PR1 & (EXTI15_10_LINES & ~(1UL << SPI_CS_PIN))) != 0U) {
    EXTI->PR1 = EXTI15_10_LINES & ~(1UL << SPI_CS_PIN);
    fault_occurred_value(FAULT_UNUSED_INTERRUPT_HANDLED, __get_IPSR());
  }
}

void llspi_init(void) {
  // one interrupt per transfer, on CS going high
  REGISTER_INTERRUPT(EXTI15_10_IRQn, EXTI15_10_IRQ_Handler, (SPI_IRQ_RATE * 2U), FAULT_INTERRUPT_RATE_SPI)

  // MOSI DMA, peripheral -> memory
  register_set(&(DMAMUX1_Channel10->CCR), 83U, 0xFFFFFFFFU);
  register_set(&(DMA2_Stream2->CR), DMA_SxCR_MINC, 0x1E077EFEU);
  register_set(&(DMA2_Stream2->PAR), (uint32_t)&(SPI4->RXDR), 0xFFFFFFFFU);

  // MISO DMA, memory -> peripheral
  register_set(&(DMAMUX1_Channel11->CCR), 84U, 0xFFFFFFFFU);
  register_set(&(DMA2_Stream3->CR), (DMA_SxCR_MINC | DMA_SxCR_DIR_0), 0x1E077EFEU);
  register_set(&(DMA2_Stream3->PAR), (uint32_t)&(SPI4->TXDR), 0xFFFFFFFFU);

  // SPI: 8-bit slave, underrun sends UDRDR
  register_set(&(SPI4->IER), 0, 0x3FFU);
  register_set(&(SPI4->CFG1), (7U << SPI_CFG1_DSIZE_Pos), SPI_CFG1_DSIZE_Msk);
  register_set(&(SPI4->UDRDR), SPI_IDLE, 0xFFFFU);
  register_set(&(SPI4->CR2), 0, 0xFFFFU);

  // CS rising edge interrupt
  register_set(&(SYSCFG->EXTICR[2]), SYSCFG_EXTICR3_EXTI11_PE, SYSCFG_EXTICR3_EXTI11);
  register_set_bits(&(EXTI->RTSR1), (1UL << SPI_CS_PIN));
  register_set_bits(&(EXTI->IMR1), (1UL << SPI_CS_PIN));
  EXTI->PR1 = (1UL << SPI_CS_PIN);
  nvic_enable(EXTI15_10_IRQn);
}
