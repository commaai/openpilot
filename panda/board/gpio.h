// this is last place with ifdef PANDA

#ifdef STM32F4
  #include "stm32f4xx_hal_gpio_ex.h"
#else
  #include "stm32f2xx_hal_gpio_ex.h"
#endif

// ********************* dynamic configuration detection *********************

#define PANDA_REV_AB 0
#define PANDA_REV_C 1

#define PULL_EFFECTIVE_DELAY 10

void puts(const char *a);

bool has_external_debug_serial = 0;
bool is_giant_panda = 0;
bool is_entering_bootmode = 0;
int revision = PANDA_REV_AB;
bool is_grey_panda = 0;

bool detect_with_pull(GPIO_TypeDef *GPIO, int pin, int mode) {
  set_gpio_mode(GPIO, pin, MODE_INPUT);
  set_gpio_pullup(GPIO, pin, mode);
  for (volatile int i=0; i<PULL_EFFECTIVE_DELAY; i++);
  bool ret = get_gpio_input(GPIO, pin);
  set_gpio_pullup(GPIO, pin, PULL_NONE);
  return ret;
}

// must call again from main because BSS is zeroed
void detect(void) {
  // detect has_external_debug_serial
  has_external_debug_serial = detect_with_pull(GPIOA, 3, PULL_DOWN);

#ifdef PANDA
  // detect is_giant_panda
  is_giant_panda = detect_with_pull(GPIOB, 1, PULL_DOWN);

  // detect panda REV C.
  // A13 floats in REV AB. In REV C, A13 is pulled up to 5V with a 10K
  // resistor and attached to the USB power control chip CTRL
  // line. Pulling A13 down with an internal 50k resistor in REV C
  // will produce a voltage divider that results in a high logic
  // level. Checking if this pin reads high with a pull down should
  // differentiate REV AB from C.
  revision = detect_with_pull(GPIOA, 13, PULL_DOWN) ? PANDA_REV_C : PANDA_REV_AB;

  // check if the ESP is trying to put me in boot mode
  is_entering_bootmode = !detect_with_pull(GPIOB, 0, PULL_UP);

  // check if it's a grey panda by seeing if the SPI lines are floating
  // TODO: is this reliable?
  is_grey_panda = !(detect_with_pull(GPIOA, 4, PULL_DOWN) | detect_with_pull(GPIOA, 5, PULL_DOWN) | detect_with_pull(GPIOA, 6, PULL_DOWN) | detect_with_pull(GPIOA, 7, PULL_DOWN));
#else
  // need to do this for early detect
  is_giant_panda = 0;
  is_grey_panda = 0;
  revision = PANDA_REV_AB;
  is_entering_bootmode = 0;
#endif
}

// ********************* bringup *********************

void periph_init(void) {
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
  RCC->APB1ENR |= RCC_APB1ENR_TIM3EN;  // slow loop and pedal
  RCC->APB1ENR |= RCC_APB1ENR_TIM4EN;  // gmlan_alt
  //RCC->APB1ENR |= RCC_APB1ENR_TIM5EN;
  //RCC->APB1ENR |= RCC_APB1ENR_TIM6EN;
  RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
  RCC->AHB2ENR |= RCC_AHB2ENR_OTGFSEN;
  //RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;
  RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;
  RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;
  RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;
}

// ********************* setters *********************

void set_can_enable(CAN_TypeDef *CAN, bool enabled) {
  // enable CAN busses
  if (CAN == CAN1) {
    #ifdef PANDA
      // CAN1_EN
      set_gpio_output(GPIOC, 1, !enabled);
    #else
      #ifdef PEDAL
        // CAN1_EN (not flipped)
        set_gpio_output(GPIOB, 3, !enabled);
      #else
        // CAN1_EN
        set_gpio_output(GPIOB, 3, enabled);
      #endif
    #endif
  } else if (CAN == CAN2) {
    #ifdef PANDA
      // CAN2_EN
      set_gpio_output(GPIOC, 13, !enabled);
    #else
      // CAN2_EN
      set_gpio_output(GPIOB, 4, enabled);
    #endif
  #ifdef CAN3
  } else if (CAN == CAN3) {
    // CAN3_EN
    set_gpio_output(GPIOA, 0, !enabled);
  #endif
  }
}

#ifdef PANDA
  #define LED_RED 9
  #define LED_GREEN 7
  #define LED_BLUE 6
#else
  #define LED_RED 10
  #define LED_GREEN 11
  #define LED_BLUE -1
#endif

void set_led(int led_num, int on) {
  if (led_num != -1) {
  #ifdef PANDA
    set_gpio_output(GPIOC, led_num, !on);
  #else
    set_gpio_output(GPIOB, led_num, !on);
  #endif
  }
}

void set_can_mode(int can, bool use_gmlan) {
  // connects to CAN2 xcvr or GMLAN xcvr
  if (use_gmlan) {
    if (can == 1) {
      // B5,B6: disable normal mode
      set_gpio_mode(GPIOB, 5, MODE_INPUT);
      set_gpio_mode(GPIOB, 6, MODE_INPUT);

      // B12,B13: gmlan mode
      set_gpio_alternate(GPIOB, 12, GPIO_AF9_CAN2);
      set_gpio_alternate(GPIOB, 13, GPIO_AF9_CAN2);
#ifdef CAN3
    } else if (can == 2) {
      // A8,A15: disable normal mode
      set_gpio_mode(GPIOA, 8, MODE_INPUT);
      set_gpio_mode(GPIOA, 15, MODE_INPUT);

      // B3,B4: enable gmlan mode
      set_gpio_alternate(GPIOB, 3, GPIO_AF11_CAN3);
      set_gpio_alternate(GPIOB, 4, GPIO_AF11_CAN3);
#endif
    }
  } else {
    if (can == 1) {
      // B12,B13: disable gmlan mode
      set_gpio_mode(GPIOB, 12, MODE_INPUT);
      set_gpio_mode(GPIOB, 13, MODE_INPUT);

      // B5,B6: normal mode
      set_gpio_alternate(GPIOB, 5, GPIO_AF9_CAN2);
      set_gpio_alternate(GPIOB, 6, GPIO_AF9_CAN2);
#ifdef CAN3
    } else if (can == 2) {
      // B3,B4: disable gmlan mode
      set_gpio_mode(GPIOB, 3, MODE_INPUT);
      set_gpio_mode(GPIOB, 4, MODE_INPUT);
      // A8,A15: normal mode
      set_gpio_alternate(GPIOA, 8, GPIO_AF11_CAN3);
      set_gpio_alternate(GPIOA, 15, GPIO_AF11_CAN3);
#endif
    }
  }
}

#define USB_POWER_NONE 0
#define USB_POWER_CLIENT 1
#define USB_POWER_CDP 2
#define USB_POWER_DCP 3

int usb_power_mode = USB_POWER_NONE;

void set_usb_power_mode(int mode) {
  bool valid_mode = true;
  switch (mode) {
    case USB_POWER_CLIENT:
      // B2,A13: set client mode
      set_gpio_output(GPIOB, 2, 0);
      set_gpio_output(GPIOA, 13, 1);
      break;
    case USB_POWER_CDP:
      // B2,A13: set CDP mode
      set_gpio_output(GPIOB, 2, 1);
      set_gpio_output(GPIOA, 13, 1);
      break;
    case USB_POWER_DCP:
      // B2,A13: set DCP mode on the charger (breaks USB!)
      set_gpio_output(GPIOB, 2, 0);
      set_gpio_output(GPIOA, 13, 0);
      break;
    default:
      valid_mode = false;
      puts("Invalid usb power mode\n");
      break;
  }

  if (valid_mode) {
    usb_power_mode = mode;
  }
}

#define ESP_DISABLED 0
#define ESP_ENABLED 1
#define ESP_BOOTMODE 2

void set_esp_mode(int mode) {
  switch (mode) {
    case ESP_DISABLED:
      // ESP OFF
      set_gpio_output(GPIOC, 14, 0);
      set_gpio_output(GPIOC, 5, 0);
      break;
    case ESP_ENABLED:
      // ESP ON
      set_gpio_output(GPIOC, 14, 1);
      set_gpio_output(GPIOC, 5, 1);
      break;
    case ESP_BOOTMODE:
      set_gpio_output(GPIOC, 14, 1);
      set_gpio_output(GPIOC, 5, 0);
      break;
    default:
      puts("Invalid esp mode\n");
      break;
  }
}

// ********************* big init function *********************

// board specific
void gpio_init(void) {
  // pull low to hold ESP in reset??
  // enable OTG out tied to ground
  GPIOA->ODR = 0;
  GPIOB->ODR = 0;
  GPIOA->PUPDR = 0;
  //GPIOC->ODR = 0;
  GPIOB->AFR[0] = 0;
  GPIOB->AFR[1] = 0;

  // C2,C3: analog mode, voltage and current sense
  set_gpio_mode(GPIOC, 2, MODE_ANALOG);
  set_gpio_mode(GPIOC, 3, MODE_ANALOG);

#ifdef PEDAL
  // comma pedal has inputs on C0 and C1
  set_gpio_mode(GPIOC, 0, MODE_ANALOG);
  set_gpio_mode(GPIOC, 1, MODE_ANALOG);
  // DAC outputs on A4 and A5
  //   apparently they don't need GPIO setup
#endif

  // C8: FAN aka TIM3_CH4
  set_gpio_alternate(GPIOC, 8, GPIO_AF2_TIM3);

  // turn off LEDs and set mode
  set_led(LED_RED, 0);
  set_led(LED_GREEN, 0);
  set_led(LED_BLUE, 0);

  // A11,A12: USB
  set_gpio_alternate(GPIOA, 11, GPIO_AF10_OTG_FS);
  set_gpio_alternate(GPIOA, 12, GPIO_AF10_OTG_FS);
  GPIOA->OSPEEDR = GPIO_OSPEEDER_OSPEEDR11 | GPIO_OSPEEDER_OSPEEDR12;

#ifdef PANDA
  // enable started_alt on the panda
  set_gpio_pullup(GPIOA, 1, PULL_UP);

  // A2,A3: USART 2 for debugging
  set_gpio_alternate(GPIOA, 2, GPIO_AF7_USART2);
  set_gpio_alternate(GPIOA, 3, GPIO_AF7_USART2);

  // A9,A10: USART 1 for talking to the ESP
  set_gpio_alternate(GPIOA, 9, GPIO_AF7_USART1);
  set_gpio_alternate(GPIOA, 10, GPIO_AF7_USART1);

  // B12: GMLAN, ignition sense, pull up
  set_gpio_pullup(GPIOB, 12, PULL_UP);

  // A4,A5,A6,A7: setup SPI
  set_gpio_alternate(GPIOA, 4, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 5, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 6, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 7, GPIO_AF5_SPI1);
#endif

  // B8,B9: CAN 1
#ifdef STM32F4
  set_gpio_alternate(GPIOB, 8, GPIO_AF8_CAN1);
  set_gpio_alternate(GPIOB, 9, GPIO_AF8_CAN1);
#else
  set_gpio_alternate(GPIOB, 8, GPIO_AF9_CAN1);
  set_gpio_alternate(GPIOB, 9, GPIO_AF9_CAN1);
#endif
  set_can_enable(CAN1, 1);

  // B5,B6: CAN 2
  set_can_mode(1, 0);
  set_can_enable(CAN2, 1);

  // A8,A15: CAN 3
  #ifdef CAN3
    set_can_mode(2, 0);
    set_can_enable(CAN3, 1);
  #endif

  /* GMLAN mode pins:
  M0(B15)  M1(B14)  mode
  =======================
  0        0        sleep
  1        0        100kbit
  0        1        high voltage wakeup
  1        1        33kbit (normal)
  */

  // put gmlan transceiver in normal mode
  set_gpio_output(GPIOB, 14, 1);
  set_gpio_output(GPIOB, 15, 1);

  #ifdef PANDA
    // K-line enable moved from B4->B7 to make room for GMLAN on CAN3
    set_gpio_output(GPIOB, 7, 1); // REV C

    // C12,D2: K-Line setup on UART 5
    set_gpio_alternate(GPIOC, 12, GPIO_AF8_UART5);
    set_gpio_alternate(GPIOD, 2, GPIO_AF8_UART5);
    set_gpio_pullup(GPIOD, 2, PULL_UP);

    // L-line enable
    set_gpio_output(GPIOA, 14, 1);

    // C10,C11: L-Line setup on USART 3
    set_gpio_alternate(GPIOC, 10, GPIO_AF7_USART3);
    set_gpio_alternate(GPIOC, 11, GPIO_AF7_USART3);
    set_gpio_pullup(GPIOC, 11, PULL_UP);
  #endif

  set_usb_power_mode(USB_POWER_CLIENT);
}

// ********************* early bringup *********************

#define ENTER_BOOTLOADER_MAGIC 0xdeadbeef
#define ENTER_SOFTLOADER_MAGIC 0xdeadc0de
#define BOOT_NORMAL 0xdeadb111

extern void *g_pfnVectors;
extern uint32_t enter_bootloader_mode;

void jump_to_bootloader(void) {
  // do enter bootloader
  enter_bootloader_mode = 0;
  void (*bootloader)(void) = (void (*)(void)) (*((uint32_t *)0x1fff0004));

  // jump to bootloader
  bootloader();

  // reset on exit
  enter_bootloader_mode = BOOT_NORMAL;
  NVIC_SystemReset();
}

void early(void) {
  // after it's been in the bootloader, things are initted differently, so we reset
  if ((enter_bootloader_mode != BOOT_NORMAL) &&
      (enter_bootloader_mode != ENTER_BOOTLOADER_MAGIC) &&
      (enter_bootloader_mode != ENTER_SOFTLOADER_MAGIC)) {
    enter_bootloader_mode = BOOT_NORMAL;
    NVIC_SystemReset();
  }

  // if wrong chip, reboot
  volatile unsigned int id = DBGMCU->IDCODE;
  #ifdef STM32F4
    if ((id&0xFFF) != 0x463) enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
  #else
    if ((id&0xFFF) != 0x411) enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
  #endif

  // setup interrupt table
  SCB->VTOR = (uint32_t)&g_pfnVectors;

  // early GPIOs float everything
  RCC->AHB1ENR = RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN | RCC_AHB1ENR_GPIOCEN;

  GPIOA->MODER = 0; GPIOB->MODER = 0; GPIOC->MODER = 0;
  GPIOA->ODR = 0; GPIOB->ODR = 0; GPIOC->ODR = 0;
  GPIOA->PUPDR = 0; GPIOB->PUPDR = 0; GPIOC->PUPDR = 0;

  detect();

  #ifdef PANDA
    // enable the ESP, disable ESP boot mode
    // unless we are on a giant panda, then there's no ESP
    // dont disable on grey panda
    if (is_giant_panda) {
      set_esp_mode(ESP_DISABLED);
    } else {
      set_esp_mode(ESP_ENABLED);
    }
  #endif


  if (enter_bootloader_mode == ENTER_BOOTLOADER_MAGIC) {
  #ifdef PANDA
    set_esp_mode(ESP_DISABLED);
  #endif
    set_led(LED_GREEN, 1);
    jump_to_bootloader();
  }

  if (is_entering_bootmode) {
    enter_bootloader_mode = ENTER_SOFTLOADER_MAGIC;
  }
}
