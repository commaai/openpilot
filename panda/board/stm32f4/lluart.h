// ***************************** Interrupt handlers *****************************

void uart_tx_ring(uart_ring *q){
  ENTER_CRITICAL();
  // Send out next byte of TX buffer
  if (q->w_ptr_tx != q->r_ptr_tx) {
    // Only send if transmit register is empty (aka last byte has been sent)
    if ((q->uart->SR & USART_SR_TXE) != 0U) {
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

// This read after reading SR clears all error interrupts. We don't want compiler warnings, nor optimizations
#define UART_READ_DR(uart) volatile uint8_t t = (uart)->DR; UNUSED(t);

// ***************************** Hardware setup *****************************

#define DIV_(_PCLK_, _BAUD_)                    (((_PCLK_) * 25U) / (4U * (_BAUD_)))
#define DIVMANT_(_PCLK_, _BAUD_)                (DIV_((_PCLK_), (_BAUD_)) / 100U)
#define DIVFRAQ_(_PCLK_, _BAUD_)                ((((DIV_((_PCLK_), (_BAUD_)) - (DIVMANT_((_PCLK_), (_BAUD_)) * 100U)) * 16U) + 50U) / 100U)
#define USART_BRR_(_PCLK_, _BAUD_)              ((DIVMANT_((_PCLK_), (_BAUD_)) << 4) | (DIVFRAQ_((_PCLK_), (_BAUD_)) & 0x0FU))

void uart_set_baud(USART_TypeDef *u, unsigned int baud) {
  u->BRR = USART_BRR_(APB1_FREQ*1000000U, baud);
}
