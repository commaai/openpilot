void spi_init() {
  puts("SPI init\n");
  SPI1->CR1 = SPI_CR1_SPE;
  SPI1->CR2 = SPI_CR2_RXNEIE | SPI_CR2_ERRIE | SPI_CR2_TXEIE;
}

void spi_tx_dma(void *addr, int len) {
  // disable DMA
  SPI1->CR2 &= ~SPI_CR2_TXDMAEN;
  DMA2_Stream3->CR &= ~DMA_SxCR_EN;

  // DMA2, stream 3, channel 3
  DMA2_Stream3->M0AR = addr;
  DMA2_Stream3->NDTR = len;
  DMA2_Stream3->PAR = &(SPI1->DR);

  // channel3, increment memory, memory -> periph, enable
  DMA2_Stream3->CR = DMA_SxCR_CHSEL_1 | DMA_SxCR_CHSEL_0 | DMA_SxCR_MINC | DMA_SxCR_DIR_0 | DMA_SxCR_EN;
  DMA2_Stream3->CR |= DMA_SxCR_TCIE;

  SPI1->CR2 |= SPI_CR2_TXDMAEN;
}

