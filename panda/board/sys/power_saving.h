#include "board/sys/sys.h"

// WARNING: To stay in compliance with the SIL2 rules laid out in STM UM2331, we should never use any of the available hardware low power modes during safety function execution.
// See rule: CoU_3

// Low power state "stop mode" is only entered from SAFETY_SILENT when no safety function is active and exited via reset which is a safe state.

bool power_save_enabled = false;
#ifdef ALLOW_DEBUG
volatile bool stop_mode_requested = false;
#endif

void enable_can_transceivers(bool enabled) {
  // Leave main CAN always on for CAN-based ignition detection
  uint8_t main_bus = (harness.status == HARNESS_STATUS_FLIPPED) ? 3U : 1U;
  for(uint8_t i=1U; i<=4U; i++){
    current_board->enable_can_transceiver(i, (i == main_bus) || enabled);
  }
}

void set_power_save_state(bool enable) {
  if (enable != power_save_enabled) {
    if (enable) {
      print("enable power savings\n");

      // Disable CAN interrupts
      if (harness.status == HARNESS_STATUS_FLIPPED) {
        llcan_irq_disable(cans[0]);
      } else {
        llcan_irq_disable(cans[2]);
      }
      llcan_irq_disable(cans[1]);
    } else {
      print("disable power savings\n");

      if (harness.status == HARNESS_STATUS_FLIPPED) {
        llcan_irq_enable(cans[0]);
      } else {
        llcan_irq_enable(cans[2]);
      }
      llcan_irq_enable(cans[1]);
    }

    enable_can_transceivers(!enable);

    // Switch off IR when in power saving
    if(enable){
      current_board->set_ir_power(0U);
    }

    power_save_enabled = enable;
  }
}

static void enter_stop_mode(void) {
  // set all GPIO to analog mode to reduce power, analog mode also disables pull resistors
  register_set(&(GPIOA->MODER), 0xFFFFFFFFU, 0xFFFFFFFFU);
  register_set(&(GPIOB->MODER), 0xFFFFFFFFU, 0xFFFFFFFFU);
  register_set(&(GPIOC->MODER), 0xFFFFFFFFU, 0xFFFFFFFFU);
  register_set(&(GPIOD->MODER), 0xFFFFFFFFU, 0xFFFFFFFFU);
  register_set(&(GPIOE->MODER), 0xFFFFFFFFU, 0xFFFFFFFFU);
  register_set(&(GPIOF->MODER), 0xFFFFFFFFU, 0xFFFFFFFFU);
  register_set(&(GPIOG->MODER), 0xFFFFFFFFU, 0xFFFFFFFFU);

  // init GPIO to lowest power state
  current_board->set_bootkick(BOOT_STANDBY);
  current_board->set_amp_enabled(false);
  for (uint8_t i = 1U; i <= 4U; i++) {
    current_board->enable_can_transceiver(i, false);
  }

  // disable ADCs
  ADC1->CR &= ~(ADC_CR_ADEN);
  ADC1->CR |= ADC_CR_DEEPPWD;
  ADC2->CR &= ~(ADC_CR_ADEN);
  ADC2->CR |= ADC_CR_DEEPPWD;

  // disable HSI48: 48 MHz USB clock
  register_clear_bits(&(RCC->CR), RCC_CR_HSI48ON);
  // disable SRAM retention in stop mode
  register_clear_bits(&(RCC->AHB2LPENR), RCC_AHB2LPENR_SRAM1LPEN | RCC_AHB2LPENR_SRAM2LPEN);
  register_clear_bits(&(RCC->AHB4LPENR), RCC_AHB4LPENR_SRAM4LPEN);
  register_clear_bits(&(RCC->AHB3LPENR), RCC_AHB3LPENR_AXISRAMLPEN);

  // SBU pins to input for EXTI wakeup
  set_gpio_mode(current_board->harness_config->GPIO_SBU1,
                current_board->harness_config->pin_SBU1, MODE_INPUT);
  set_gpio_mode(current_board->harness_config->GPIO_SBU2,
                current_board->harness_config->pin_SBU2, MODE_INPUT);

  // EXTI1: SBU2 (PA1)
  // EXTI4: SBU1 (PC4)
  register_set(&(SYSCFG->EXTICR[0]), SYSCFG_EXTICR1_EXTI1_PA, 0xF0U);
  register_set(&(SYSCFG->EXTICR[1]), SYSCFG_EXTICR2_EXTI4_PC, 0xFU);
  register_set_bits(&(EXTI->IMR1), (1U << 1) | (1U << 4));
  register_set_bits(&(EXTI->RTSR1), (1U << 1) | (1U << 4));
  register_set_bits(&(EXTI->FTSR1), (1U << 1) | (1U << 4));

  // EXTI for CAN wakeup
  // EXTI8:  FDCAN1 RX (PB8)
  // EXTI5:  FDCAN2 RX (PB5)
  // EXTI12: FDCAN3 RX (PD12)
  set_gpio_mode(GPIOB, 8, MODE_INPUT);
  register_set(&(SYSCFG->EXTICR[2]), SYSCFG_EXTICR3_EXTI8_PB, 0xFU);
  set_gpio_mode(GPIOB, 5, MODE_INPUT);
  register_set(&(SYSCFG->EXTICR[1]), SYSCFG_EXTICR2_EXTI5_PB, 0xF0U);
  set_gpio_mode(GPIOD, 12, MODE_INPUT);
  register_set(&(SYSCFG->EXTICR[3]), SYSCFG_EXTICR4_EXTI12_PD, 0xFU);
  uint32_t can_exti_line = (1UL << 8) | (1UL << 5) | (1UL << 12);
  register_set_bits(&(EXTI->IMR1), can_exti_line);
  register_set_bits(&(EXTI->FTSR1), can_exti_line);

  // clear pending EXTI
  EXTI->PR1 = (1U << 1) | (1U << 4) | can_exti_line;

  // reset if ignition just came on before going to sleep
  if (harness_check_ignition()) {
    NVIC_SystemReset();
  }

  // stop mode
  register_clear_bits(&(PWR->CPUCR), PWR_CPUCR_PDDS_D1 | PWR_CPUCR_PDDS_D2 | PWR_CPUCR_PDDS_D3);

  // set SVOS5 voltage scaling, flash low-power
  register_set(&(PWR->CR1), PWR_CR1_SVOS_0 | PWR_CR1_FLPS, PWR_CR1_SVOS | PWR_CR1_FLPS);

  // enter stop mode on WFI
  SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;

  __disable_irq();

  // disable all NVIC interrupts and clear pending
  for (uint32_t i = 0U; i < 8U; i++) {
    NVIC->ICER[i] = 0xFFFFFFFFU;
    NVIC->ICPR[i] = 0xFFFFFFFFU;
  }
  // enable only wakeup EXTI interrupts
  NVIC_EnableIRQ(EXTI1_IRQn);     // SBU2 (PA1)
  NVIC_EnableIRQ(EXTI4_IRQn);     // SBU1 (PC4)
  NVIC_EnableIRQ(EXTI9_5_IRQn);    // FDCAN1 RX (PB8), FDCAN2 RX (PB5)
  NVIC_EnableIRQ(EXTI15_10_IRQn);  // FDCAN3 RX (PD12)

  __DSB();
  __ISB();
  __WFI();

  NVIC_SystemReset();
}
