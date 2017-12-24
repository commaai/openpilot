// IRQs: USART1, USART2, USART3, UART5

// ***************************** serial port queues *****************************

// esp = USART1
uart_ring esp_ring = { .w_ptr_tx = 0, .r_ptr_tx = 0,
                       .w_ptr_rx = 0, .r_ptr_rx = 0,
                       .uart = USART1 };

// lin1, K-LINE = UART5
// lin2, L-LINE = USART3
uart_ring lin1_ring = { .w_ptr_tx = 0, .r_ptr_tx = 0,
                        .w_ptr_rx = 0, .r_ptr_rx = 0,
                        .uart = UART5 };
uart_ring lin2_ring = { .w_ptr_tx = 0, .r_ptr_tx = 0,
                        .w_ptr_rx = 0, .r_ptr_rx = 0,
                        .uart = USART3 };

// debug = USART2
void debug_ring_callback(uart_ring *ring);
uart_ring debug_ring = { .w_ptr_tx = 0, .r_ptr_tx = 0,
                         .w_ptr_rx = 0, .r_ptr_rx = 0,
                         .uart = USART2,
                         .callback = debug_ring_callback};


uart_ring *get_ring_by_number(int a) {
  switch(a) {
    case 0:
      return &debug_ring;
    case 1:
      return &esp_ring;
    case 2:
      return &lin1_ring;
    case 3:
      return &lin2_ring;
    default:
      return NULL;
  }
}

// ***************************** serial port *****************************

void uart_ring_process(uart_ring *q) {
  enter_critical_section();
  // TODO: check if external serial is connected
  int sr = q->uart->SR;

  if (q->w_ptr_tx != q->r_ptr_tx) {
    if (sr & USART_SR_TXE) {
      q->uart->DR = q->elems_tx[q->r_ptr_tx];
      q->r_ptr_tx += 1;
    } else {
      // push on interrupt later
      q->uart->CR1 |= USART_CR1_TXEIE;
    }
  } else {
    // nothing to send
    q->uart->CR1 &= ~USART_CR1_TXEIE;
  }

  if (sr & USART_SR_RXNE) {
    uint8_t c = q->uart->DR;  // TODO: can drop packets
    uint8_t next_w_ptr = q->w_ptr_rx + 1;
    if (next_w_ptr != q->r_ptr_rx) {
      q->elems_rx[q->w_ptr_rx] = c;
      q->w_ptr_rx = next_w_ptr;
      if (q->callback) q->callback(q);
    }
  }
  exit_critical_section();
}

// interrupt boilerplate

void USART1_IRQHandler(void) { uart_ring_process(&esp_ring); }
void USART2_IRQHandler(void) { uart_ring_process(&debug_ring); }
void USART3_IRQHandler(void) { uart_ring_process(&lin2_ring); }
void UART5_IRQHandler(void) { uart_ring_process(&lin1_ring); }

int getc(uart_ring *q, char *elem) {
  int ret = 0;

  enter_critical_section();
  if (q->w_ptr_rx != q->r_ptr_rx) {
    *elem = q->elems_rx[q->r_ptr_rx];
    q->r_ptr_rx += 1;
    ret = 1;
  }
  exit_critical_section();

  return ret;
}

int injectc(uart_ring *q, char elem) {
  int ret = 0;
  uint8_t next_w_ptr;

  enter_critical_section();
  next_w_ptr = q->w_ptr_rx + 1;
  if (next_w_ptr != q->r_ptr_rx) {
    q->elems_rx[q->w_ptr_rx] = elem;
    q->w_ptr_rx = next_w_ptr;
    ret = 1;
  }
  exit_critical_section();

  return ret;
}

int putc(uart_ring *q, char elem) {
  int ret = 0;
  uint8_t next_w_ptr;

  enter_critical_section();
  next_w_ptr = q->w_ptr_tx + 1;
  if (next_w_ptr != q->r_ptr_tx) {
    q->elems_tx[q->w_ptr_tx] = elem;
    q->w_ptr_tx = next_w_ptr;
    ret = 1;
  }
  exit_critical_section();

  uart_ring_process(q);

  return ret;
}

void clear_uart_buff(uart_ring *q) {
  enter_critical_section();
  q->w_ptr_tx = 0;
  q->r_ptr_tx = 0;
  q->w_ptr_rx = 0;
  q->r_ptr_rx = 0;
  exit_critical_section();
}

// ***************************** start UART code *****************************

#define __DIV(_PCLK_, _BAUD_)                        (((_PCLK_)*25)/(4*(_BAUD_)))
#define __DIVMANT(_PCLK_, _BAUD_)                    (__DIV((_PCLK_), (_BAUD_))/100)
#define __DIVFRAQ(_PCLK_, _BAUD_)                    (((__DIV((_PCLK_), (_BAUD_)) - (__DIVMANT((_PCLK_), (_BAUD_)) * 100)) * 16 + 50) / 100)
#define __USART_BRR(_PCLK_, _BAUD_)              ((__DIVMANT((_PCLK_), (_BAUD_)) << 4)|(__DIVFRAQ((_PCLK_), (_BAUD_)) & 0x0F))

void uart_set_baud(USART_TypeDef *u, int baud) {
  if (u == USART1) {
    // USART1 is on APB2
    u->BRR = __USART_BRR(48000000, baud);
  } else {
    u->BRR = __USART_BRR(24000000, baud);
  }
}

void uart_init(USART_TypeDef *u, int baud) {
  // enable uart and tx+rx mode
  u->CR1 = USART_CR1_UE;
  uart_set_baud(u, baud);

  u->CR1 |= USART_CR1_TE | USART_CR1_RE;
  //u->CR2 = USART_CR2_STOP_0 | USART_CR2_STOP_1;
  //u->CR2 = USART_CR2_STOP_0;
  // ** UART is ready to work **

  // enable interrupts
  u->CR1 |= USART_CR1_RXNEIE;

  if (u == USART1) {
    NVIC_EnableIRQ(USART1_IRQn);
  } else if (u == USART2) {
    NVIC_EnableIRQ(USART2_IRQn);
  } else if (u == USART3) {
    NVIC_EnableIRQ(USART3_IRQn);
  } else if (u == UART5) {
    NVIC_EnableIRQ(UART5_IRQn);
  }
}

void putch(const char a) {
  if (has_external_debug_serial) {
    /*while ((debug_ring.uart->SR & USART_SR_TXE) == 0);
    debug_ring.uart->DR = a;*/

    // assuming debugging is important if there's external serial connected
    while (!putc(&debug_ring, a));

    //putc(&debug_ring, a);
  } else {
    injectc(&debug_ring, a);
  }
}

int puts(const char *a) {
  for (;*a;a++) {
    if (*a == '\n') putch('\r');
    putch(*a);
  }
  return 0;
}

void puth(unsigned int i) {
  int pos;
  char c[] = "0123456789abcdef";
  for (pos = 28; pos != -4; pos -= 4) {
    putch(c[(i >> pos) & 0xF]);
  }
}

void puth2(unsigned int i) {
  int pos;
  char c[] = "0123456789abcdef";
  for (pos = 4; pos != -4; pos -= 4) {
    putch(c[(i >> pos) & 0xF]);
  }
}

void hexdump(const void *a, int l) {
  int i;
  for (i=0;i<l;i++) {
    if (i != 0 && (i&0xf) == 0) puts("\n");
    puth2(((const unsigned char*)a)[i]);
    puts(" ");
  }
  puts("\n");
}
