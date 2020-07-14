#ifdef STM32F4
  #include "stm32f4xx_hal_gpio_ex.h"
#else
  #include "stm32f2xx_hal_gpio_ex.h"
#endif

// Common GPIO initialization
void common_init_gpio(void){
  // TODO: Is this block actually doing something???
  // pull low to hold ESP in reset??
  // enable OTG out tied to ground
  GPIOA->ODR = 0;
  GPIOB->ODR = 0;
  GPIOA->PUPDR = 0;
  GPIOB->AFR[0] = 0;
  GPIOB->AFR[1] = 0;

  // C2: Voltage sense line
  set_gpio_mode(GPIOC, 2, MODE_ANALOG);

  // A11,A12: USB
  set_gpio_alternate(GPIOA, 11, GPIO_AF10_OTG_FS);
  set_gpio_alternate(GPIOA, 12, GPIO_AF10_OTG_FS);
  GPIOA->OSPEEDR = GPIO_OSPEEDER_OSPEEDR11 | GPIO_OSPEEDER_OSPEEDR12;

  // A9,A10: USART 1 for talking to the ESP / GPS
  set_gpio_alternate(GPIOA, 9, GPIO_AF7_USART1);
  set_gpio_alternate(GPIOA, 10, GPIO_AF7_USART1);

   // B8,B9: CAN 1
  #ifdef STM32F4
    set_gpio_alternate(GPIOB, 8, GPIO_AF8_CAN1);
    set_gpio_alternate(GPIOB, 9, GPIO_AF8_CAN1);
  #else
    set_gpio_alternate(GPIOB, 8, GPIO_AF9_CAN1);
    set_gpio_alternate(GPIOB, 9, GPIO_AF9_CAN1);
  #endif
}

// Peripheral initialization
void peripherals_init(void){
  // enable GPIOB, UART2, CAN, USB clock
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOCEN;
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;

  RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;
  RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
  RCC->APB1ENR |= RCC_APB1ENR_USART3EN;
  #ifdef PANDA
    RCC->APB1ENR |= RCC_APB1ENR_UART5EN;
  #endif
  RCC->APB1ENR |= RCC_APB1ENR_CAN1EN;
  RCC->APB1ENR |= RCC_APB1ENR_CAN2EN;
  #ifdef CAN3
    RCC->APB1ENR |= RCC_APB1ENR_CAN3EN;
  #endif
  RCC->APB1ENR |= RCC_APB1ENR_DACEN;
  RCC->APB1ENR |= RCC_APB1ENR_TIM2EN;  // main counter
  RCC->APB1ENR |= RCC_APB1ENR_TIM3EN;  // pedal and fan PWM
  RCC->APB1ENR |= RCC_APB1ENR_TIM4EN;  // gmlan_alt and IR PWM
  RCC->APB1ENR |= RCC_APB1ENR_TIM5EN;  // k-line init
  RCC->APB1ENR |= RCC_APB1ENR_TIM6EN;  // interrupt timer
  RCC->APB1ENR |= RCC_APB1ENR_PWREN;   // for RTC config
  RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
  RCC->AHB2ENR |= RCC_AHB2ENR_OTGFSEN;
  //RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;
  RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;
  RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;
  RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;
  RCC->APB2ENR |= RCC_APB2ENR_TIM9EN;  // slow loop
}

// Detection with internal pullup
#define PULL_EFFECTIVE_DELAY 4096
bool detect_with_pull(GPIO_TypeDef *GPIO, int pin, int mode) {
  set_gpio_mode(GPIO, pin, MODE_INPUT);
  set_gpio_pullup(GPIO, pin, mode);
  for (volatile int i=0; i<PULL_EFFECTIVE_DELAY; i++);
  bool ret = get_gpio_input(GPIO, pin);
  set_gpio_pullup(GPIO, pin, PULL_NONE);
  return ret;
}