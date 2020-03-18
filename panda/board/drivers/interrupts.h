typedef struct interrupt {
  IRQn_Type irq_type;
  void (*handler)(void);
  uint32_t call_counter;
  uint32_t max_call_rate;   // Call rate is defined as the amount of calls each second
  uint32_t call_rate_fault;
} interrupt;

void unused_interrupt_handler(void) {
  // Something is wrong if this handler is called!
  puts("Unused interrupt handler called!\n");
  fault_occurred(FAULT_UNUSED_INTERRUPT_HANDLED);
}

#define NUM_INTERRUPTS 102U                // There are 102 external interrupt sources (see stm32f413.h)
interrupt interrupts[NUM_INTERRUPTS];

#define REGISTER_INTERRUPT(irq_num, func_ptr, call_rate, rate_fault) \
  interrupts[irq_num].irq_type = irq_num; \
  interrupts[irq_num].handler = func_ptr;  \
  interrupts[irq_num].call_counter = 0U;   \
  interrupts[irq_num].max_call_rate = call_rate; \
  interrupts[irq_num].call_rate_fault = rate_fault;

bool check_interrupt_rate = false;

void handle_interrupt(IRQn_Type irq_type){
  interrupts[irq_type].call_counter++;
  interrupts[irq_type].handler();

  // Check that the interrupts don't fire too often
  if(check_interrupt_rate && (interrupts[irq_type].call_counter > interrupts[irq_type].max_call_rate)){
    puts("Interrupt 0x"); puth(irq_type); puts(" fired too often (0x"); puth(interrupts[irq_type].call_counter); puts("/s)!\n");
    fault_occurred(interrupts[irq_type].call_rate_fault);
  }
}

// Reset interrupt counter every second
void TIM6_DAC_IRQ_Handler(void) {
  if (TIM6->SR != 0) {
    for(uint16_t i=0U; i<NUM_INTERRUPTS; i++){
      interrupts[i].call_counter = 0U;
    }
  }
  TIM6->SR = 0;
}

void init_interrupts(bool check_rate_limit){
  check_interrupt_rate = check_rate_limit;

  for(uint16_t i=0U; i<NUM_INTERRUPTS; i++){
    interrupts[i].handler = unused_interrupt_handler;
  }

  // Init timer 10 for a 1s interval
  register_set_bits(&(RCC->APB1ENR), RCC_APB1ENR_TIM6EN);  // Enable interrupt timer peripheral
  REGISTER_INTERRUPT(TIM6_DAC_IRQn, TIM6_DAC_IRQ_Handler, 1, FAULT_INTERRUPT_RATE_INTERRUPTS)
  register_set(&(TIM6->PSC), (732-1), 0xFFFFU);
  register_set(&(TIM6->DIER), TIM_DIER_UIE, 0x5F5FU);
  register_set(&(TIM6->CR1), TIM_CR1_CEN, 0x3FU);
  TIM6->SR = 0;
  NVIC_EnableIRQ(TIM6_DAC_IRQn);
}

// ********************* Bare interrupt handlers *********************
// Only implemented the STM32F413 interrupts for now, the STM32F203 specific ones do not fall into the scope of SIL2

void WWDG_IRQHandler(void) {handle_interrupt(WWDG_IRQn);}
void PVD_IRQHandler(void) {handle_interrupt(PVD_IRQn);}
void TAMP_STAMP_IRQHandler(void) {handle_interrupt(TAMP_STAMP_IRQn);}
void RTC_WKUP_IRQHandler(void) {handle_interrupt(RTC_WKUP_IRQn);}
void FLASH_IRQHandler(void) {handle_interrupt(FLASH_IRQn);}
void RCC_IRQHandler(void) {handle_interrupt(RCC_IRQn);}
void EXTI0_IRQHandler(void) {handle_interrupt(EXTI0_IRQn);}
void EXTI1_IRQHandler(void) {handle_interrupt(EXTI1_IRQn);}
void EXTI2_IRQHandler(void) {handle_interrupt(EXTI2_IRQn);}
void EXTI3_IRQHandler(void) {handle_interrupt(EXTI3_IRQn);}
void EXTI4_IRQHandler(void) {handle_interrupt(EXTI4_IRQn);}
void DMA1_Stream0_IRQHandler(void) {handle_interrupt(DMA1_Stream0_IRQn);}
void DMA1_Stream1_IRQHandler(void) {handle_interrupt(DMA1_Stream1_IRQn);}
void DMA1_Stream2_IRQHandler(void) {handle_interrupt(DMA1_Stream2_IRQn);}
void DMA1_Stream3_IRQHandler(void) {handle_interrupt(DMA1_Stream3_IRQn);}
void DMA1_Stream4_IRQHandler(void) {handle_interrupt(DMA1_Stream4_IRQn);}
void DMA1_Stream5_IRQHandler(void) {handle_interrupt(DMA1_Stream5_IRQn);}
void DMA1_Stream6_IRQHandler(void) {handle_interrupt(DMA1_Stream6_IRQn);}
void ADC_IRQHandler(void) {handle_interrupt(ADC_IRQn);}
void CAN1_TX_IRQHandler(void) {handle_interrupt(CAN1_TX_IRQn);}
void CAN1_RX0_IRQHandler(void) {handle_interrupt(CAN1_RX0_IRQn);}
void CAN1_RX1_IRQHandler(void) {handle_interrupt(CAN1_RX1_IRQn);}
void CAN1_SCE_IRQHandler(void) {handle_interrupt(CAN1_SCE_IRQn);}
void EXTI9_5_IRQHandler(void) {handle_interrupt(EXTI9_5_IRQn);}
void TIM1_BRK_TIM9_IRQHandler(void) {handle_interrupt(TIM1_BRK_TIM9_IRQn);}
void TIM1_UP_TIM10_IRQHandler(void) {handle_interrupt(TIM1_UP_TIM10_IRQn);}
void TIM1_TRG_COM_TIM11_IRQHandler(void) {handle_interrupt(TIM1_TRG_COM_TIM11_IRQn);}
void TIM1_CC_IRQHandler(void) {handle_interrupt(TIM1_CC_IRQn);}
void TIM2_IRQHandler(void) {handle_interrupt(TIM2_IRQn);}
void TIM3_IRQHandler(void) {handle_interrupt(TIM3_IRQn);}
void TIM4_IRQHandler(void) {handle_interrupt(TIM4_IRQn);}
void I2C1_EV_IRQHandler(void) {handle_interrupt(I2C1_EV_IRQn);}
void I2C1_ER_IRQHandler(void) {handle_interrupt(I2C1_ER_IRQn);}
void I2C2_EV_IRQHandler(void) {handle_interrupt(I2C2_EV_IRQn);}
void I2C2_ER_IRQHandler(void) {handle_interrupt(I2C2_ER_IRQn);}
void SPI1_IRQHandler(void) {handle_interrupt(SPI1_IRQn);}
void SPI2_IRQHandler(void) {handle_interrupt(SPI2_IRQn);}
void USART1_IRQHandler(void) {handle_interrupt(USART1_IRQn);}
void USART2_IRQHandler(void) {handle_interrupt(USART2_IRQn);}
void USART3_IRQHandler(void) {handle_interrupt(USART3_IRQn);}
void EXTI15_10_IRQHandler(void) {handle_interrupt(EXTI15_10_IRQn);}
void RTC_Alarm_IRQHandler(void) {handle_interrupt(RTC_Alarm_IRQn);}
void OTG_FS_WKUP_IRQHandler(void) {handle_interrupt(OTG_FS_WKUP_IRQn);}
void TIM8_BRK_TIM12_IRQHandler(void) {handle_interrupt(TIM8_BRK_TIM12_IRQn);}
void TIM8_UP_TIM13_IRQHandler(void) {handle_interrupt(TIM8_UP_TIM13_IRQn);}
void TIM8_TRG_COM_TIM14_IRQHandler(void) {handle_interrupt(TIM8_TRG_COM_TIM14_IRQn);}
void TIM8_CC_IRQHandler(void) {handle_interrupt(TIM8_CC_IRQn);}
void DMA1_Stream7_IRQHandler(void) {handle_interrupt(DMA1_Stream7_IRQn);}
void FSMC_IRQHandler(void) {handle_interrupt(FSMC_IRQn);}
void SDIO_IRQHandler(void) {handle_interrupt(SDIO_IRQn);}
void TIM5_IRQHandler(void) {handle_interrupt(TIM5_IRQn);}
void SPI3_IRQHandler(void) {handle_interrupt(SPI3_IRQn);}
void UART4_IRQHandler(void) {handle_interrupt(UART4_IRQn);}
void UART5_IRQHandler(void) {handle_interrupt(UART5_IRQn);}
void TIM6_DAC_IRQHandler(void) {handle_interrupt(TIM6_DAC_IRQn);}
void TIM7_IRQHandler(void) {handle_interrupt(TIM7_IRQn);}
void DMA2_Stream0_IRQHandler(void) {handle_interrupt(DMA2_Stream0_IRQn);}
void DMA2_Stream1_IRQHandler(void) {handle_interrupt(DMA2_Stream1_IRQn);}
void DMA2_Stream2_IRQHandler(void) {handle_interrupt(DMA2_Stream2_IRQn);}
void DMA2_Stream3_IRQHandler(void) {handle_interrupt(DMA2_Stream3_IRQn);}
void DMA2_Stream4_IRQHandler(void) {handle_interrupt(DMA2_Stream4_IRQn);}
void CAN2_TX_IRQHandler(void) {handle_interrupt(CAN2_TX_IRQn);}
void CAN2_RX0_IRQHandler(void) {handle_interrupt(CAN2_RX0_IRQn);}
void CAN2_RX1_IRQHandler(void) {handle_interrupt(CAN2_RX1_IRQn);}
void CAN2_SCE_IRQHandler(void) {handle_interrupt(CAN2_SCE_IRQn);}
void OTG_FS_IRQHandler(void) {handle_interrupt(OTG_FS_IRQn);}
void DMA2_Stream5_IRQHandler(void) {handle_interrupt(DMA2_Stream5_IRQn);}
void DMA2_Stream6_IRQHandler(void) {handle_interrupt(DMA2_Stream6_IRQn);}
void DMA2_Stream7_IRQHandler(void) {handle_interrupt(DMA2_Stream7_IRQn);}
void USART6_IRQHandler(void) {handle_interrupt(USART6_IRQn);}
void I2C3_EV_IRQHandler(void) {handle_interrupt(I2C3_EV_IRQn);}
void I2C3_ER_IRQHandler(void) {handle_interrupt(I2C3_ER_IRQn);}
#ifdef STM32F4
  void DFSDM1_FLT0_IRQHandler(void) {handle_interrupt(DFSDM1_FLT0_IRQn);}
  void DFSDM1_FLT1_IRQHandler(void) {handle_interrupt(DFSDM1_FLT1_IRQn);}
  void CAN3_TX_IRQHandler(void) {handle_interrupt(CAN3_TX_IRQn);}
  void CAN3_RX0_IRQHandler(void) {handle_interrupt(CAN3_RX0_IRQn);}
  void CAN3_RX1_IRQHandler(void) {handle_interrupt(CAN3_RX1_IRQn);}
  void CAN3_SCE_IRQHandler(void) {handle_interrupt(CAN3_SCE_IRQn);}
  void RNG_IRQHandler(void) {handle_interrupt(RNG_IRQn);}
  void FPU_IRQHandler(void) {handle_interrupt(FPU_IRQn);}
  void UART7_IRQHandler(void) {handle_interrupt(UART7_IRQn);}
  void UART8_IRQHandler(void) {handle_interrupt(UART8_IRQn);}
  void SPI4_IRQHandler(void) {handle_interrupt(SPI4_IRQn);}
  void SPI5_IRQHandler(void) {handle_interrupt(SPI5_IRQn);}
  void SAI1_IRQHandler(void) {handle_interrupt(SAI1_IRQn);}
  void UART9_IRQHandler(void) {handle_interrupt(UART9_IRQn);}
  void UART10_IRQHandler(void) {handle_interrupt(UART10_IRQn);}
  void QUADSPI_IRQHandler(void) {handle_interrupt(QUADSPI_IRQn);}
  void FMPI2C1_EV_IRQHandler(void) {handle_interrupt(FMPI2C1_EV_IRQn);}
  void FMPI2C1_ER_IRQHandler(void) {handle_interrupt(FMPI2C1_ER_IRQn);}
  void LPTIM1_IRQHandler(void) {handle_interrupt(LPTIM1_IRQn);}
  void DFSDM2_FLT0_IRQHandler(void) {handle_interrupt(DFSDM2_FLT0_IRQn);}
  void DFSDM2_FLT1_IRQHandler(void) {handle_interrupt(DFSDM2_FLT1_IRQn);}
  void DFSDM2_FLT2_IRQHandler(void) {handle_interrupt(DFSDM2_FLT2_IRQn);}
  void DFSDM2_FLT3_IRQHandler(void) {handle_interrupt(DFSDM2_FLT3_IRQn);}
#endif