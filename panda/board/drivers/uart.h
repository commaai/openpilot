// IRQs: USART1, USART2, USART3, UART5

// ***************************** Definitions *****************************
#define FIFO_SIZE_INT 0x400U
#define FIFO_SIZE_DMA 0x1000U

typedef struct uart_ring {
  volatile uint16_t w_ptr_tx;
  volatile uint16_t r_ptr_tx;
  uint8_t *elems_tx;
  uint32_t tx_fifo_size;
  volatile uint16_t w_ptr_rx;
  volatile uint16_t r_ptr_rx;
  uint8_t *elems_rx;
  uint32_t rx_fifo_size;
  USART_TypeDef *uart;
  void (*callback)(struct uart_ring*);
  bool dma_rx;
} uart_ring;

#define UART_BUFFER(x, size_rx, size_tx, uart_ptr, callback_ptr, rx_dma) \
  uint8_t elems_rx_##x[size_rx]; \
  uint8_t elems_tx_##x[size_tx]; \
  uart_ring uart_ring_##x = {  \
    .w_ptr_tx = 0, \
    .r_ptr_tx = 0, \
    .elems_tx = ((uint8_t *)&elems_tx_##x), \
    .tx_fifo_size = size_tx, \
    .w_ptr_rx = 0, \
    .r_ptr_rx = 0, \
    .elems_rx = ((uint8_t *)&elems_rx_##x), \
    .rx_fifo_size = size_rx, \
    .uart = uart_ptr, \
    .callback = callback_ptr, \
    .dma_rx = rx_dma \
  };


// ***************************** Function prototypes *****************************
void uart_init(uart_ring *q, int baud);

bool getc(uart_ring *q, char *elem);
bool putc(uart_ring *q, char elem);

void puts(const char *a);
void puth(unsigned int i);
void hexdump(const void *a, int l);

void debug_ring_callback(uart_ring *ring);

// ******************************** UART buffers ********************************

// esp_gps = USART1
UART_BUFFER(esp_gps, FIFO_SIZE_DMA, FIFO_SIZE_INT, USART1, NULL, true)

// lin1, K-LINE = UART5
// lin2, L-LINE = USART3
UART_BUFFER(lin1, FIFO_SIZE_INT, FIFO_SIZE_INT, UART5, NULL, false)
UART_BUFFER(lin2, FIFO_SIZE_INT, FIFO_SIZE_INT, USART3, NULL, false)

// debug = USART2
UART_BUFFER(debug, FIFO_SIZE_INT, FIFO_SIZE_INT, USART2, debug_ring_callback, false)

uart_ring *get_ring_by_number(int a) {
  uart_ring *ring = NULL;
  switch(a) {
    case 0:
      ring = &uart_ring_debug;
      break;
    case 1:
      ring = &uart_ring_esp_gps;
      break;
    case 2:
      ring = &uart_ring_lin1;
      break;
    case 3:
      ring = &uart_ring_lin2;
      break;
    default:
      ring = NULL;
      break;
  }
  return ring;
}

// ***************************** Interrupt handlers *****************************

void uart_tx_ring(uart_ring *q){
  ENTER_CRITICAL();
  // Send out next byte of TX buffer
  if (q->w_ptr_tx != q->r_ptr_tx) {
    // Only send if transmit register is empty (aka last byte has been sent)
    if ((q->uart->SR & USART_SR_TXE) != 0) {
      q->uart->DR = q->elems_tx[q->r_ptr_tx];   // This clears TXE
      q->r_ptr_tx = (q->r_ptr_tx + 1U) % q->tx_fifo_size;
    }

    // Enable TXE interrupt if there is still data to be sent
    if(q->r_ptr_tx != q->w_ptr_tx){
      q->uart->CR1 |= USART_CR1_TXEIE;
    } else {
      q->uart->CR1 &= ~USART_CR1_TXEIE;
    }
  }
  EXIT_CRITICAL();
}

void uart_rx_ring(uart_ring *q){
  // Do not read out directly if DMA enabled
  if (q->dma_rx == false) {
    ENTER_CRITICAL();

    // Read out RX buffer
    uint8_t c = q->uart->DR;  // This read after reading SR clears a bunch of interrupts

    uint16_t next_w_ptr = (q->w_ptr_rx + 1U) % q->rx_fifo_size;
    // Do not overwrite buffer data
    if (next_w_ptr != q->r_ptr_rx) {
      q->elems_rx[q->w_ptr_rx] = c;
      q->w_ptr_rx = next_w_ptr;
      if (q->callback != NULL) {
        q->callback(q);
      }
    }

    EXIT_CRITICAL();
  }
}

// This function should be called on:
// * Half-transfer DMA interrupt
// * Full-transfer DMA interrupt
// * UART IDLE detection
uint32_t prev_w_index = 0;
void dma_pointer_handler(uart_ring *q, uint32_t dma_ndtr) {
  ENTER_CRITICAL();
  uint32_t w_index = (q->rx_fifo_size - dma_ndtr);
  
  // Check for new data
  if (w_index != prev_w_index){
    // Check for overflow
    if (
      ((prev_w_index < q->r_ptr_rx) && (q->r_ptr_rx <= w_index)) ||                               // No rollover
      ((w_index < prev_w_index) && ((q->r_ptr_rx <= w_index) || (prev_w_index < q->r_ptr_rx)))    // Rollover
    ){   
      // We lost data. Set the new read pointer to the oldest byte still available
      q->r_ptr_rx = (w_index + 1U) % q->rx_fifo_size;
    }

    // Set write pointer
    q->w_ptr_rx = w_index;
  }

  prev_w_index = w_index;
  EXIT_CRITICAL();
}

// This read after reading SR clears all error interrupts. We don't want compiler warnings, nor optimizations
#define UART_READ_DR(uart) volatile uint8_t t = (uart)->DR; UNUSED(t);

void uart_interrupt_handler(uart_ring *q) {
  ENTER_CRITICAL();

  // Read UART status. This is also the first step necessary in clearing most interrupts
  uint32_t status = q->uart->SR;

  // If RXNE is set, perform a read. This clears RXNE, ORE, IDLE, NF and FE
  if((status & USART_SR_RXNE) != 0U){
    uart_rx_ring(q);
  }

  // Detect errors and clear them
  uint32_t err = (status & USART_SR_ORE) | (status & USART_SR_NE) | (status & USART_SR_FE) | (status & USART_SR_PE);
  if(err != 0U){
    #ifdef DEBUG_UART
      puts("Encountered UART error: "); puth(err); puts("\n");
    #endif
    UART_READ_DR(q->uart)
  }
  // Send if necessary
  uart_tx_ring(q);

  // Run DMA pointer handler if the line is idle
  if(q->dma_rx && (status & USART_SR_IDLE)){
    // Reset IDLE flag
    UART_READ_DR(q->uart)

    if(q == &uart_ring_esp_gps){
      dma_pointer_handler(&uart_ring_esp_gps, DMA2_Stream5->NDTR);
    } else {
      #ifdef DEBUG_UART
        puts("No IDLE dma_pointer_handler implemented for this UART.");
      #endif
    }
  }

  EXIT_CRITICAL();
}

void USART1_IRQHandler(void) { uart_interrupt_handler(&uart_ring_esp_gps); }
void USART2_IRQHandler(void) { uart_interrupt_handler(&uart_ring_debug); }
void USART3_IRQHandler(void) { uart_interrupt_handler(&uart_ring_lin2); }
void UART5_IRQHandler(void) { uart_interrupt_handler(&uart_ring_lin1); }

void DMA2_Stream5_IRQHandler(void) {
  ENTER_CRITICAL();

  // Handle errors
  if((DMA2->HISR & DMA_HISR_TEIF5) || (DMA2->HISR & DMA_HISR_DMEIF5) || (DMA2->HISR & DMA_HISR_FEIF5)){
    #ifdef DEBUG_UART
      puts("Encountered UART DMA error. Clearing and restarting DMA...\n");
    #endif

    // Clear flags
    DMA2->HIFCR = DMA_HIFCR_CTEIF5 | DMA_HIFCR_CDMEIF5 | DMA_HIFCR_CFEIF5;

    // Re-enable the DMA if necessary
    DMA2_Stream5->CR |= DMA_SxCR_EN;
  }

  // Re-calculate write pointer and reset flags
  dma_pointer_handler(&uart_ring_esp_gps, DMA2_Stream5->NDTR);
  DMA2->HIFCR = DMA_HIFCR_CTCIF5 | DMA_HIFCR_CHTIF5;

  EXIT_CRITICAL();
}

// ***************************** Hardware setup *****************************

void dma_rx_init(uart_ring *q) {
  // Initialization is UART-dependent
  if(q == &uart_ring_esp_gps){
    // DMA2, stream 5, channel 4

    // Disable FIFO mode (enable direct)
    DMA2_Stream5->FCR &= ~DMA_SxFCR_DMDIS;

    // Setup addresses
    DMA2_Stream5->PAR = (uint32_t)&(USART1->DR);    // Source
    DMA2_Stream5->M0AR = (uint32_t)q->elems_rx;     // Destination
    DMA2_Stream5->NDTR = q->rx_fifo_size;           // Number of bytes to copy

    // Circular, Increment memory, byte size, periph -> memory, enable
    // Transfer complete, half transfer, transfer error and direct mode error interrupt enable
    DMA2_Stream5->CR = DMA_SxCR_CHSEL_2 | DMA_SxCR_MINC | DMA_SxCR_CIRC | DMA_SxCR_HTIE | DMA_SxCR_TCIE | DMA_SxCR_TEIE | DMA_SxCR_DMEIE | DMA_SxCR_EN;
    
    // Enable DMA receiver in UART
    q->uart->CR3 |= USART_CR3_DMAR;

    // Enable UART IDLE interrupt
    q->uart->CR1 |= USART_CR1_IDLEIE;

    // Enable interrupt
    NVIC_EnableIRQ(DMA2_Stream5_IRQn);
  } else {
    puts("Tried to initialize RX DMA for an unsupported UART\n");
  }
}

#define __DIV(_PCLK_, _BAUD_)                    (((_PCLK_) * 25U) / (4U * (_BAUD_)))
#define __DIVMANT(_PCLK_, _BAUD_)                (__DIV((_PCLK_), (_BAUD_)) / 100U)
#define __DIVFRAQ(_PCLK_, _BAUD_)                ((((__DIV((_PCLK_), (_BAUD_)) - (__DIVMANT((_PCLK_), (_BAUD_)) * 100U)) * 16U) + 50U) / 100U)
#define __USART_BRR(_PCLK_, _BAUD_)              ((__DIVMANT((_PCLK_), (_BAUD_)) << 4) | (__DIVFRAQ((_PCLK_), (_BAUD_)) & 0x0FU))

void uart_set_baud(USART_TypeDef *u, unsigned int baud) {
  if (u == USART1) {
    // USART1 is on APB2
    u->BRR = __USART_BRR(48000000U, baud);
  } else {
    u->BRR = __USART_BRR(24000000U, baud);
  }
}

void uart_init(uart_ring *q, int baud) {
  // Set baud and enable peripheral with TX and RX mode
  uart_set_baud(q->uart, baud);
  q->uart->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;

  // Enable UART interrupts
  if(q->uart == USART1){
    NVIC_EnableIRQ(USART1_IRQn);
  } else if (q->uart == USART2){
    NVIC_EnableIRQ(USART2_IRQn);
  } else if (q->uart == USART3){
    NVIC_EnableIRQ(USART3_IRQn);
  } else if (q->uart == UART5){
    NVIC_EnableIRQ(UART5_IRQn);
  } else {
    // UART not used. Skip enabling interrupts
  }

  // Initialise RX DMA if used
  if(q->dma_rx){
    dma_rx_init(q);
  }
}

// ************************* Low-level buffer functions *************************

bool getc(uart_ring *q, char *elem) {
  bool ret = false;

  ENTER_CRITICAL();
  if (q->w_ptr_rx != q->r_ptr_rx) {
    if (elem != NULL) *elem = q->elems_rx[q->r_ptr_rx];
    q->r_ptr_rx = (q->r_ptr_rx + 1U) % q->rx_fifo_size;
    ret = true;
  }
  EXIT_CRITICAL();

  return ret;
}

bool injectc(uart_ring *q, char elem) {
  int ret = false;
  uint16_t next_w_ptr;

  ENTER_CRITICAL();
  next_w_ptr = (q->w_ptr_rx + 1U) % q->tx_fifo_size;
  if (next_w_ptr != q->r_ptr_rx) {
    q->elems_rx[q->w_ptr_rx] = elem;
    q->w_ptr_rx = next_w_ptr;
    ret = true;
  }
  EXIT_CRITICAL();

  return ret;
}

bool putc(uart_ring *q, char elem) {
  bool ret = false;
  uint16_t next_w_ptr;

  ENTER_CRITICAL();
  next_w_ptr = (q->w_ptr_tx + 1U) % q->tx_fifo_size;
  if (next_w_ptr != q->r_ptr_tx) {
    q->elems_tx[q->w_ptr_tx] = elem;
    q->w_ptr_tx = next_w_ptr;
    ret = true;
  }
  EXIT_CRITICAL();

  uart_tx_ring(q);

  return ret;
}

// Seems dangerous to use (might lock CPU if called with interrupts disabled f.e.)
// TODO: Remove? Not used anyways
void uart_flush(uart_ring *q) {
  while (q->w_ptr_tx != q->r_ptr_tx) {
    __WFI();
  }
}

void uart_flush_sync(uart_ring *q) {
  // empty the TX buffer
  while (q->w_ptr_tx != q->r_ptr_tx) {
    uart_tx_ring(q);
  }
}

void uart_send_break(uart_ring *u) {
  while ((u->uart->CR1 & USART_CR1_SBK) != 0);
  u->uart->CR1 |= USART_CR1_SBK;
}

void clear_uart_buff(uart_ring *q) {
  ENTER_CRITICAL();
  q->w_ptr_tx = 0;
  q->r_ptr_tx = 0;
  q->w_ptr_rx = 0;
  q->r_ptr_rx = 0;
  EXIT_CRITICAL();
}

// ************************ High-level debug functions **********************
void putch(const char a) {
  if (has_external_debug_serial) {
    // assuming debugging is important if there's external serial connected
    while (!putc(&uart_ring_debug, a));

  } else {
    // misra-c2012-17.7: serial debug function, ok to ignore output
    (void)injectc(&uart_ring_debug, a);
  }
}

void puts(const char *a) {
  for (const char *in = a; *in; in++) {
    if (*in == '\n') putch('\r');
    putch(*in);
  }
}

void putui(uint32_t i) {
  uint32_t i_copy = i;
  char str[11];
  uint8_t idx = 10;
  str[idx] = '\0';
  idx--;
  do {
    str[idx] = (i_copy % 10U) + 0x30U;
    idx--;
    i_copy /= 10;
  } while (i_copy != 0U);
  puts(&str[idx + 1U]);
}

void puth(unsigned int i) {
  char c[] = "0123456789abcdef";
  for (int pos = 28; pos != -4; pos -= 4) {
    putch(c[(i >> (unsigned int)(pos)) & 0xFU]);
  }
}

void puth2(unsigned int i) {
  char c[] = "0123456789abcdef";
  for (int pos = 4; pos != -4; pos -= 4) {
    putch(c[(i >> (unsigned int)(pos)) & 0xFU]);
  }
}

void hexdump(const void *a, int l) {
  if (a != NULL) {
    for (int i=0; i < l; i++) {
      if ((i != 0) && ((i & 0xf) == 0)) puts("\n");
      puth2(((const unsigned char*)a)[i]);
      puts(" ");
    }
  }
  puts("\n");
}
