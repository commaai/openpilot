// flasher state variables
uint32_t *prog_ptr = NULL;
int unlocked = 0;

#ifdef uart_ring
void debug_ring_callback(uart_ring *ring) {}
#endif

int usb_cb_control_msg(USB_Setup_TypeDef *setup, uint8_t *resp, bool hardwired) {
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
      current_board->set_led(LED_GREEN, 1);
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
      #ifdef STM32F4
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
          // TODO: put this back when it's no longer a "devkit"
          //#ifdef ALLOW_DEBUG
          #if 1
          if (hardwired) {
          #else
          // no more bootstub on UNO once OTP block is flashed
          if (hardwired && ((hw_type != HW_TYPE_UNO) || (!is_provisioned()))) {
          #endif
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
      COMPILE_TIME_ASSERT(sizeof(gitversion) <= MAX_RESP_LEN);
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

int usb_cb_ep1_in(void *usbdata, int len, bool hardwired) {
  UNUSED(usbdata);
  UNUSED(len);
  UNUSED(hardwired);
  return 0;
}
void usb_cb_ep3_out(void *usbdata, int len, bool hardwired) {
  UNUSED(usbdata);
  UNUSED(len);
  UNUSED(hardwired);
}
void usb_cb_ep3_out_complete(void) {}

int is_enumerated = 0;
void usb_cb_enumeration_complete(void) {
  puts("USB enumeration complete\n");
  is_enumerated = 1;
}

void usb_cb_ep2_out(void *usbdata, int len, bool hardwired) {
  UNUSED(hardwired);
  current_board->set_led(LED_RED, 0);
  for (int i = 0; i < len/4; i++) {
    // program byte 1
    FLASH->CR = FLASH_CR_PSIZE_1 | FLASH_CR_PG;

    *prog_ptr = *(uint32_t*)(usbdata+(i*4));
    while (FLASH->SR & FLASH_SR_BSY);

    //*(uint64_t*)(&spi_tx_buf[0x30+(i*4)]) = *prog_ptr;
    prog_ptr++;
  }
  current_board->set_led(LED_RED, 1);
}


int spi_cb_rx(uint8_t *data, int len, uint8_t *data_out) {
  UNUSED(len);
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

#ifdef PEDAL

#include "stm32fx/llcan.h"
#define CAN CAN1

#define CAN_BL_INPUT 0x1
#define CAN_BL_OUTPUT 0x2

void CAN1_TX_IRQ_Handler(void) {
  // clear interrupt
  CAN->TSR |= CAN_TSR_RQCP0;
}

#define ISOTP_BUF_SIZE 0x110

uint8_t isotp_buf[ISOTP_BUF_SIZE];
uint8_t *isotp_buf_ptr = NULL;
int isotp_buf_remain = 0;

uint8_t isotp_buf_out[ISOTP_BUF_SIZE];
uint8_t *isotp_buf_out_ptr = NULL;
int isotp_buf_out_remain = 0;
int isotp_buf_out_idx = 0;

void bl_can_send(uint8_t *odat) {
  // wait for send
  while (!(CAN->TSR & CAN_TSR_TME0));

  // send continue
  CAN->sTxMailBox[0].TDLR = ((uint32_t*)odat)[0];
  CAN->sTxMailBox[0].TDHR = ((uint32_t*)odat)[1];
  CAN->sTxMailBox[0].TDTR = 8;
  CAN->sTxMailBox[0].TIR = (CAN_BL_OUTPUT << 21) | 1;
}

void CAN1_RX0_IRQ_Handler(void) {
  while (CAN->RF0R & CAN_RF0R_FMP0) {
    if ((CAN->sFIFOMailBox[0].RIR>>21) == CAN_BL_INPUT) {
      uint8_t dat[8];
      for (int i = 0; i < 8; i++) {
        dat[i] = GET_BYTE(&CAN->sFIFOMailBox[0], i);
      }
      uint8_t odat[8];
      uint8_t type = dat[0] & 0xF0;
      if (type == 0x30) {
        // continue
        while (isotp_buf_out_remain > 0) {
          // wait for send
          while (!(CAN->TSR & CAN_TSR_TME0));

          odat[0] = 0x20 | isotp_buf_out_idx;
          memcpy(odat+1, isotp_buf_out_ptr, 7);
          isotp_buf_out_remain -= 7;
          isotp_buf_out_ptr += 7;
          isotp_buf_out_idx++;

          bl_can_send(odat);
        }
      } else if (type == 0x20) {
        if (isotp_buf_remain > 0) {
          memcpy(isotp_buf_ptr, dat+1, 7);
          isotp_buf_ptr += 7;
          isotp_buf_remain -= 7;
        }
        if (isotp_buf_remain <= 0) {
          int len = isotp_buf_ptr - isotp_buf + isotp_buf_remain;

          // call the function
          memset(isotp_buf_out, 0, ISOTP_BUF_SIZE);
          isotp_buf_out_remain = spi_cb_rx(isotp_buf, len, isotp_buf_out);
          isotp_buf_out_ptr = isotp_buf_out;
          isotp_buf_out_idx = 0;

          // send initial
          if (isotp_buf_out_remain <= 7) {
            odat[0] = isotp_buf_out_remain;
            memcpy(odat+1, isotp_buf_out_ptr, isotp_buf_out_remain);
          } else {
            odat[0] = 0x10 | (isotp_buf_out_remain>>8);
            odat[1] = isotp_buf_out_remain & 0xFF;
            memcpy(odat+2, isotp_buf_out_ptr, 6);
            isotp_buf_out_remain -= 6;
            isotp_buf_out_ptr += 6;
            isotp_buf_out_idx++;
          }

          bl_can_send(odat);
        }
      } else if (type == 0x10) {
        int len = ((dat[0]&0xF)<<8) | dat[1];

        // setup buffer
        isotp_buf_ptr = isotp_buf;
        memcpy(isotp_buf_ptr, dat+2, 6);

        if (len < (ISOTP_BUF_SIZE-0x10)) {
          isotp_buf_ptr += 6;
          isotp_buf_remain = len-6;
        }

        memset(odat, 0, 8);
        odat[0] = 0x30;
        bl_can_send(odat);
      }
    }
    // next
    CAN->RF0R |= CAN_RF0R_RFOM0;
  }
}

void CAN1_SCE_IRQ_Handler(void) {
  llcan_clear_send(CAN);
}

#endif

void soft_flasher_start(void) {
  #ifdef PEDAL
    REGISTER_INTERRUPT(CAN1_TX_IRQn, CAN1_TX_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_1)
    REGISTER_INTERRUPT(CAN1_RX0_IRQn, CAN1_RX0_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_1)
    REGISTER_INTERRUPT(CAN1_SCE_IRQn, CAN1_SCE_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_1)
  #endif

  puts("\n\n\n************************ FLASHER START ************************\n");

  enter_bootloader_mode = 0;

  RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;
  RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;
  RCC->AHB2ENR |= RCC_AHB2ENR_OTGFSEN;
  RCC->APB1ENR |= RCC_APB1ENR_USART2EN;

// pedal has the canloader
#ifdef PEDAL
  RCC->APB1ENR |= RCC_APB1ENR_CAN1EN;

  // B8,B9: CAN 1
  set_gpio_alternate(GPIOB, 8, GPIO_AF9_CAN1);
  set_gpio_alternate(GPIOB, 9, GPIO_AF9_CAN1);
  current_board->enable_can_transceiver(1, true);

  // init can
  llcan_set_speed(CAN1, 5000, false, false);
  llcan_init(CAN1);
#endif

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
  //spi_init();

  // enable USB
  usb_init();

  // green LED on for flashing
  current_board->set_led(LED_GREEN, 1);

  enable_interrupts();

  uint64_t cnt = 0;

  for (cnt=0;;cnt++) {
    if (cnt == 35 && !is_enumerated && usb_power_mode == USB_POWER_CLIENT) {
      // if you are connected through a hub to the phone
      // you need power to be able to see the device
      puts("USBP: didn't enumerate, switching to CDP mode\n");
      current_board->set_usb_power_mode(USB_POWER_CDP);
      current_board->set_led(LED_BLUE, 1);
    }
    // blink the green LED fast
    current_board->set_led(LED_GREEN, 0);
    delay(500000);
    current_board->set_led(LED_GREEN, 1);
    delay(500000);
  }
}
