// flasher state variables
uint32_t *prog_ptr = NULL;
int unlocked = 0;

void debug_ring_callback(uart_ring *ring) {}

int usb_cb_control_msg(USB_Setup_TypeDef *setup, uint8_t *resp, int hardwired) {
  int resp_len = 0;

  // flasher machine
  memset(resp, 0, 4);
  memcpy(resp+4, "\xde\xad\xd0\x0d", 4);
  resp[0] = 0xff;
  resp[2] = setup->b.bRequest;
  resp[3] = ~setup->b.bRequest;
  *((uint32_t **)&resp[8]) = prog_ptr;
  resp_len = 0xc;

  int sec;
  switch (setup->b.bRequest) {
    // **** 0xb0: flasher echo
    case 0xb0:
      resp[1] = 0xff;
      break;
    // **** 0xb1: unlock flash
    case 0xb1:
      if (FLASH->CR & FLASH_CR_LOCK) {
        FLASH->KEYR = 0x45670123;
        FLASH->KEYR = 0xCDEF89AB;
        resp[1] = 0xff;
      }
      set_led(LED_GREEN, 1);
      unlocked = 1;
      prog_ptr = (uint32_t *)0x8004000;
      break;
    // **** 0xb2: erase sector
    case 0xb2:
      sec = setup->b.wValue.w;
      // don't erase the bootloader
      if (sec != 0 && sec < 12 && unlocked) {
        FLASH->CR = (sec << 3) | FLASH_CR_SER;
        FLASH->CR |= FLASH_CR_STRT;
        while (FLASH->SR & FLASH_SR_BSY);
        resp[1] = 0xff;
      }
      break;
    // **** 0xd0: fetch serial number
    case 0xd0:
      #ifdef PANDA
        // addresses are OTP
        if (setup->b.wValue.w == 1) {
          memcpy(resp, (void *)0x1fff79c0, 0x10);
          resp_len = 0x10;
        } else {
          get_provision_chunk(resp);
          resp_len = PROVISION_CHUNK_LEN;
        }
      #endif
      break;
    // **** 0xd1: enter bootloader mode
    case 0xd1:
      // this allows reflashing of the bootstub
      // so it's blocked over wifi
      switch (setup->b.wValue.w) {
        case 0:
          if (hardwired) {
            puts("-> entering bootloader\n");
            enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
            NVIC_SystemReset();
          }
          break;
        case 1:
          puts("-> entering softloader\n");
          enter_bootloader_mode = ENTER_SOFTLOADER_MAGIC;
          NVIC_SystemReset();
          break;
      }
      break;
    // **** 0xd6: get version
    case 0xd6:
      COMPILE_TIME_ASSERT(sizeof(gitversion) <= MAX_RESP_LEN)
      memcpy(resp, gitversion, sizeof(gitversion));
      resp_len = sizeof(gitversion);
      break;
    // **** 0xd8: reset ST
    case 0xd8:
      NVIC_SystemReset();
      break;
  }
  return resp_len;
}

int usb_cb_ep1_in(uint8_t *usbdata, int len, int hardwired) { return 0; }
void usb_cb_ep3_out(uint8_t *usbdata, int len, int hardwired) { }

int is_enumerated = 0;
void usb_cb_enumeration_complete() {
  puts("USB enumeration complete\n");
  is_enumerated = 1;
}

void usb_cb_ep2_out(uint8_t *usbdata, int len, int hardwired) {
  set_led(LED_RED, 0);
  for (int i = 0; i < len/4; i++) {
    // program byte 1
    FLASH->CR = FLASH_CR_PSIZE_1 | FLASH_CR_PG;

    *prog_ptr = *(uint32_t*)(usbdata+(i*4));
    while (FLASH->SR & FLASH_SR_BSY);

    //*(uint64_t*)(&spi_tx_buf[0x30+(i*4)]) = *prog_ptr;
    prog_ptr++;
  }
  set_led(LED_RED, 1);
}


int spi_cb_rx(uint8_t *data, int len, uint8_t *data_out) {
  int resp_len = 0;
  switch (data[0]) {
    case 0:
      // control transfer
      resp_len = usb_cb_control_msg((USB_Setup_TypeDef *)(data+4), data_out, 0);
      break;
    case 2:
      // ep 2, flash!
      usb_cb_ep2_out(data+4, data[2], 0);
      break;
  }
  return resp_len;
}

void soft_flasher_start() {
  puts("\n\n\n************************ FLASHER START ************************\n");

  enter_bootloader_mode = 0;

  RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;
  RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;
  RCC->AHB2ENR |= RCC_AHB2ENR_OTGFSEN;
  RCC->APB1ENR |= RCC_APB1ENR_USART2EN;

  // A4,A5,A6,A7: setup SPI
  set_gpio_alternate(GPIOA, 4, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 5, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 6, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 7, GPIO_AF5_SPI1);

  // A2,A3: USART 2 for debugging
  set_gpio_alternate(GPIOA, 2, GPIO_AF7_USART2);
  set_gpio_alternate(GPIOA, 3, GPIO_AF7_USART2);

  // A11,A12: USB
  set_gpio_alternate(GPIOA, 11, GPIO_AF10_OTG_FS);
  set_gpio_alternate(GPIOA, 12, GPIO_AF10_OTG_FS);
  GPIOA->OSPEEDR = GPIO_OSPEEDER_OSPEEDR11 | GPIO_OSPEEDER_OSPEEDR12;

  // flasher
  spi_init();

  // enable USB
  usb_init();

  // green LED on for flashing
  set_led(LED_GREEN, 1);

  __enable_irq();

  uint64_t cnt = 0;

  for (cnt=0;;cnt++) {
    if (cnt == 35 && !is_enumerated && usb_power_mode == USB_POWER_CLIENT) {
      // if you are connected through a hub to the phone
      // you need power to be able to see the device
      puts("USBP: didn't enumerate, switching to CDP mode\n");
      set_usb_power_mode(USB_POWER_CDP);
      set_led(LED_BLUE, 1);
    }
    // blink the green LED fast
    set_led(LED_GREEN, 0);
    delay(500000);
    set_led(LED_GREEN, 1);
    delay(500000);
  }
}

