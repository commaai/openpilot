// flasher state variables
uint32_t *prog_ptr = NULL;
bool unlocked = false;

void spi_init(void);

#ifdef uart_ring
void debug_ring_callback(uart_ring *ring) {}
#endif

int comms_control_handler(ControlPacket_t *req, uint8_t *resp) {
  int resp_len = 0;

  // flasher machine
  memset(resp, 0, 4);
  memcpy(resp+4, "\xde\xad\xd0\x0d", 4);
  resp[0] = 0xff;
  resp[2] = req->request;
  resp[3] = ~req->request;
  *((uint32_t **)&resp[8]) = prog_ptr;
  resp_len = 0xc;

  int sec;
  switch (req->request) {
    // **** 0xb0: flasher echo
    case 0xb0:
      resp[1] = 0xff;
      break;
    // **** 0xb1: unlock flash
    case 0xb1:
      if (flash_is_locked()) {
        flash_unlock();
        resp[1] = 0xff;
      }
      current_board->set_led(LED_GREEN, 1);
      unlocked = true;
      prog_ptr = (uint32_t *)APP_START_ADDRESS;
      break;
    // **** 0xb2: erase sector
    case 0xb2:
      sec = req->param1;
      if (flash_erase_sector(sec, unlocked)) {
        resp[1] = 0xff;
      }
      break;
    // **** 0xc1: get hardware type
    case 0xc1:
      resp[0] = hw_type;
      resp_len = 1;
      break;
    // **** 0xc3: fetch MCU UID
    case 0xc3:
      #ifdef UID_BASE
        (void)memcpy(resp, ((uint8_t *)UID_BASE), 12);
        resp_len = 12;
      #endif
      break;
    // **** 0xd0: fetch serial number
    case 0xd0:
      #ifndef STM32F2
        // addresses are OTP
        if (req->param1 == 1) {
          memcpy(resp, (void *)DEVICE_SERIAL_NUMBER_ADDRESS, 0x10);
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
      switch (req->param1) {
        case 0:
          print("-> entering bootloader\n");
          enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
          NVIC_SystemReset();
          break;
        case 1:
          print("-> entering softloader\n");
          enter_bootloader_mode = ENTER_SOFTLOADER_MAGIC;
          NVIC_SystemReset();
          break;
      }
      break;
    // **** 0xd6: get version
    case 0xd6:
      COMPILE_TIME_ASSERT(sizeof(gitversion) <= USBPACKET_MAX_SIZE);
      memcpy(resp, gitversion, sizeof(gitversion));
      resp_len = sizeof(gitversion);
      break;
    // **** 0xd8: reset ST
    case 0xd8:
      flush_write_buffer();
      NVIC_SystemReset();
      break;
  }
  return resp_len;
}

void comms_can_write(uint8_t *data, uint32_t len) {
  UNUSED(data);
  UNUSED(len);
}

int comms_can_read(uint8_t *data, uint32_t max_len) {
  UNUSED(data);
  UNUSED(max_len);
  return 0;
}

void usb_cb_ep3_out_complete(void) {}

void comms_endpoint2_write(uint8_t *data, uint32_t len) {
  current_board->set_led(LED_RED, 0);
  for (uint32_t i = 0; i < len/4; i++) {
    flash_write_word(prog_ptr, *(uint32_t*)(data+(i*4)));

    //*(uint64_t*)(&spi_tx_buf[0x30+(i*4)]) = *prog_ptr;
    prog_ptr++;
  }
  current_board->set_led(LED_RED, 1);
}


int spi_cb_rx(uint8_t *data, int len, uint8_t *data_out) {
  UNUSED(len);
  ControlPacket_t control_req;

  int resp_len = 0;
  switch (data[0]) {
    case 0:
      // control transfer
      control_req.request = ((USB_Setup_TypeDef *)(data+4))->b.bRequest;
      control_req.param1 = ((USB_Setup_TypeDef *)(data+4))->b.wValue.w;
      control_req.param2 = ((USB_Setup_TypeDef *)(data+4))->b.wIndex.w;
      control_req.length = ((USB_Setup_TypeDef *)(data+4))->b.wLength.w;

      resp_len = comms_control_handler(&control_req, data_out);
      break;
    case 2:
      // ep 2, flash!
      comms_endpoint2_write(data+4, data[2]);
      break;
  }
  return resp_len;
}

#ifdef PEDAL

#include "stm32fx/llbxcan.h"
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
        dat[i] = GET_MAILBOX_BYTE(&CAN->sFIFOMailBox[0], i);
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

  print("\n\n\n************************ FLASHER START ************************\n");

  enter_bootloader_mode = 0;

  flasher_peripherals_init();

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

  gpio_usart2_init();
  gpio_usb_init();

  // enable USB
  usb_init();

  // enable SPI
  if (current_board->has_spi) {
    gpio_spi_init();
    spi_init();
  }

  // green LED on for flashing
  current_board->set_led(LED_GREEN, 1);

  enable_interrupts();

  for (;;) {
    // blink the green LED fast
    current_board->set_led(LED_GREEN, 0);
    delay(500000);
    current_board->set_led(LED_GREEN, 1);
    delay(500000);
  }
}
