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
    .elems_tx = ((uint8_t *)&(elems_tx_##x)), \
    .tx_fifo_size = (size_tx), \
    .w_ptr_rx = 0, \
    .r_ptr_rx = 0, \
    .elems_rx = ((uint8_t *)&(elems_rx_##x)), \
    .rx_fifo_size = (size_rx), \
    .uart = (uart_ptr), \
    .callback = (callback_ptr), \
    .dma_rx = (rx_dma) \
  };

// ***************************** Function prototypes *****************************
void debug_ring_callback(uart_ring *ring);
void uart_tx_ring(uart_ring *q);
void uart_send_break(uart_ring *u);

// ******************************** UART buffers ********************************

// gps = USART1
UART_BUFFER(gps, FIFO_SIZE_DMA, FIFO_SIZE_INT, USART1, NULL, true)

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
      ring = &uart_ring_gps;
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

void puthx(uint32_t i, uint8_t len) {
  const char c[] = "0123456789abcdef";
  for (int pos = ((int)len * 4) - 4; pos > -4; pos -= 4) {
    putch(c[(i >> (unsigned int)(pos)) & 0xFU]);
  }
}

void puth(unsigned int i) {
  puthx(i, 8U);
}

void puth2(unsigned int i) {
  puthx(i, 2U);
}

void puth4(unsigned int i) {
  puthx(i, 4U);
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
