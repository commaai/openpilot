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
  ENTER_CRITICAL();

  // Read out RX buffer
  uint8_t c = q->uart->DR;  // This read after reading SR clears a bunch of interrupts

  uint16_t next_w_ptr = (q->w_ptr_rx + 1U) % q->rx_fifo_size;

  if ((next_w_ptr == q->r_ptr_rx) && q->overwrite) {
    // overwrite mode: drop oldest byte
    q->r_ptr_rx = (q->r_ptr_rx + 1U) % q->rx_fifo_size;
  }

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

void uart_send_break(uart_ring *u) {
  while ((u->uart->CR1 & USART_CR1_SBK) != 0);
  u->uart->CR1 |= USART_CR1_SBK;
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
      print("Encountered UART error: "); puth(err); print("\n");
    #endif
    UART_READ_DR(q->uart)
  }
  // Send if necessary
  uart_tx_ring(q);

  EXIT_CRITICAL();
}

void USART2_IRQ_Handler(void) { uart_interrupt_handler(&uart_ring_debug); }
void USART3_IRQ_Handler(void) { uart_interrupt_handler(&uart_ring_lin2); }
void UART5_IRQ_Handler(void) { uart_interrupt_handler(&uart_ring_lin1); }

// ***************************** Hardware setup *****************************

#define __DIV(_PCLK_, _BAUD_)                    (((_PCLK_) * 25U) / (4U * (_BAUD_)))
#define __DIVMANT(_PCLK_, _BAUD_)                (__DIV((_PCLK_), (_BAUD_)) / 100U)
#define __DIVFRAQ(_PCLK_, _BAUD_)                ((((__DIV((_PCLK_), (_BAUD_)) - (__DIVMANT((_PCLK_), (_BAUD_)) * 100U)) * 16U) + 50U) / 100U)
#define __USART_BRR(_PCLK_, _BAUD_)              ((__DIVMANT((_PCLK_), (_BAUD_)) << 4) | (__DIVFRAQ((_PCLK_), (_BAUD_)) & 0x0FU))

void uart_set_baud(USART_TypeDef *u, unsigned int baud) {
  u->BRR = __USART_BRR(APB1_FREQ*1000000U, baud);
}

void uart_init(uart_ring *q, int baud) {
  if(q->uart != NULL){
    // Register interrupts (max data rate: 115200 baud)
    if (q->uart == USART2){
      REGISTER_INTERRUPT(USART2_IRQn, USART2_IRQ_Handler, 150000U, FAULT_INTERRUPT_RATE_UART_2)
    } else if (q->uart == USART3){
      REGISTER_INTERRUPT(USART3_IRQn, USART3_IRQ_Handler, 150000U, FAULT_INTERRUPT_RATE_UART_3)
    } else if (q->uart == UART5){
      REGISTER_INTERRUPT(UART5_IRQn, UART5_IRQ_Handler, 150000U, FAULT_INTERRUPT_RATE_UART_5)
    } else {
      // UART not used. Skip registering interrupts
    }

    // Set baud and enable peripheral with TX and RX mode
    uart_set_baud(q->uart, baud);
    q->uart->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
    if ((q->uart == USART2) || (q->uart == USART3) || (q->uart == UART5)) {
      q->uart->CR1 |= USART_CR1_RXNEIE;
    }

    // Enable UART interrupts
    if (q->uart == USART2){
      NVIC_EnableIRQ(USART2_IRQn);
    } else if (q->uart == USART3){
      NVIC_EnableIRQ(USART3_IRQn);
    } else if (q->uart == UART5){
      NVIC_EnableIRQ(UART5_IRQn);
    } else {
      // UART not used. Skip enabling interrupts
    }
  }
}
