
void dma_pointer_handler(uart_ring *q, uint32_t dma_ndtr) { UNUSED(q); UNUSED(dma_ndtr); }
void dma_rx_init(uart_ring *q) { UNUSED(q); }

#define __DIV(_PCLK_, _BAUD_)                    (((_PCLK_) * 25U) / (4U * (_BAUD_)))
#define __DIVMANT(_PCLK_, _BAUD_)                (__DIV((_PCLK_), (_BAUD_)) / 100U)
#define __DIVFRAQ(_PCLK_, _BAUD_)                ((((__DIV((_PCLK_), (_BAUD_)) - (__DIVMANT((_PCLK_), (_BAUD_)) * 100U)) * 16U) + 50U) / 100U)
#define __USART_BRR(_PCLK_, _BAUD_)              ((__DIVMANT((_PCLK_), (_BAUD_)) << 4) | (__DIVFRAQ((_PCLK_), (_BAUD_)) & 0x0FU))

void uart_rx_ring(uart_ring *q){
  // Do not read out directly if DMA enabled
  if (q->dma_rx == false) {
    ENTER_CRITICAL();

    // Read out RX buffer
    uint8_t c = q->uart->RDR;  // This read after reading SR clears a bunch of interrupts

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
}

void uart_tx_ring(uart_ring *q){
  ENTER_CRITICAL();
  // Send out next byte of TX buffer
  if (q->w_ptr_tx != q->r_ptr_tx) {
    // Only send if transmit register is empty (aka last byte has been sent)
    if ((q->uart->ISR & USART_ISR_TXE_TXFNF) != 0) {
      q->uart->TDR = q->elems_tx[q->r_ptr_tx];   // This clears TXE
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

void uart_set_baud(USART_TypeDef *u, unsigned int baud) {
  // UART7 is connected to APB1 at 60MHz
  u->BRR = 60000000U / baud;
}

// This read after reading ISR clears all error interrupts. We don't want compiler warnings, nor optimizations
#define UART_READ_RDR(uart) volatile uint8_t t = (uart)->RDR; UNUSED(t);

void uart_interrupt_handler(uart_ring *q) {
  ENTER_CRITICAL();

  // Read UART status. This is also the first step necessary in clearing most interrupts
  uint32_t status = q->uart->ISR;

  // If RXFNE is set, perform a read. This clears RXFNE, ORE, IDLE, NF and FE
  if((status & USART_ISR_RXNE_RXFNE) != 0U){
    uart_rx_ring(q);
  }

  // Detect errors and clear them
  uint32_t err = (status & USART_ISR_ORE) | (status & USART_ISR_NE) | (status & USART_ISR_FE) | (status & USART_ISR_PE);
  if(err != 0U){
    #ifdef DEBUG_UART
      print("Encountered UART error: "); puth(err); print("\n");
    #endif
    UART_READ_RDR(q->uart)
  }

  if ((err & USART_ISR_ORE) != 0U) {
    q->uart->ICR |= USART_ICR_ORECF;
  } else if ((err & USART_ISR_NE) != 0U) {
    q->uart->ICR |= USART_ICR_NECF;
  } else if ((err & USART_ISR_FE) != 0U) {
    q->uart->ICR |= USART_ICR_FECF;
  } else if ((err & USART_ISR_PE) != 0U) {
    q->uart->ICR |= USART_ICR_PECF;
  } else {}

  // Send if necessary
  uart_tx_ring(q);

  // Run DMA pointer handler if the line is idle
  if(q->dma_rx && (status & USART_ISR_IDLE)){
    // Reset IDLE flag
    UART_READ_RDR(q->uart)

    #ifdef DEBUG_UART
      print("No IDLE dma_pointer_handler implemented for this UART.");
    #endif
  }

  EXIT_CRITICAL();
}

void UART7_IRQ_Handler(void) { uart_interrupt_handler(&uart_ring_som_debug); }

void uart_init(uart_ring *q, int baud) {
  if (q->uart == UART7) {
    REGISTER_INTERRUPT(UART7_IRQn, UART7_IRQ_Handler, 150000U, FAULT_INTERRUPT_RATE_UART_7)

    if (q->dma_rx) {
      // TODO
    }

    uart_set_baud(q->uart, baud);
    q->uart->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;

    // Enable interrupt on RX not empty
    q->uart->CR1 |= USART_CR1_RXNEIE;

    // Enable UART interrupts
    NVIC_EnableIRQ(UART7_IRQn);

    // Initialise RX DMA if used
    if (q->dma_rx) {
      dma_rx_init(q);
    }
  }
}
