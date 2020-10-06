// Early bringup
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
  // Reset global critical depth
  global_critical_depth = 0;

  // Init register and interrupt tables
  init_registers();

  // neccesary for DFU flashing on a non-power cycled white panda
  enable_interrupts();

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
    if ((id & 0xFFFU) != 0x463U) {
      enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
    }
  #else
    if ((id & 0xFFFU) != 0x411U) {
      enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
    }
  #endif

  // setup interrupt table
  SCB->VTOR = (uint32_t)&g_pfnVectors;

  // early GPIOs float everything
  RCC->AHB1ENR = RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN | RCC_AHB1ENR_GPIOCEN;

  GPIOA->MODER = 0; GPIOB->MODER = 0; GPIOC->MODER = 0;
  GPIOA->ODR = 0; GPIOB->ODR = 0; GPIOC->ODR = 0;
  GPIOA->PUPDR = 0; GPIOB->PUPDR = 0; GPIOC->PUPDR = 0;

  detect_configuration();
  detect_board_type();

  if (enter_bootloader_mode == ENTER_BOOTLOADER_MAGIC) {
  #ifdef PANDA
    current_board->set_gps_mode(GPS_DISABLED);
  #endif
    current_board->set_led(LED_GREEN, 1);
    jump_to_bootloader();
  }
}
