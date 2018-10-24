#include "config.h"
#include "obj/gitversion.h"

// ********************* includes *********************

#include "libc.h"
#include "safety.h"
#include "provision.h"

#include "drivers/drivers.h"

#include "drivers/llgpio.h"
#include "gpio.h"

#include "drivers/uart.h"
#include "drivers/adc.h"
#include "drivers/usb.h"
#include "drivers/gmlan_alt.h"
#include "drivers/can.h"
#include "drivers/spi.h"
#include "drivers/timer.h"


// ***************************** fan *****************************

void fan_init() {
  // timer for fan PWM
  TIM3->CCMR2 = TIM_CCMR2_OC3M_2 | TIM_CCMR2_OC3M_1;
  TIM3->CCER = TIM_CCER_CC3E;
  timer_init(TIM3, 10);
}

void fan_set_speed(int fan_speed) {
  TIM3->CCR3 = fan_speed;
}

// ********************* serial debugging *********************

void debug_ring_callback(uart_ring *ring) {
  char rcv;
  while (getc(ring, &rcv)) {
    putc(ring, rcv);

    // jump to DFU flash
    if (rcv == 'z') {
      enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
      NVIC_SystemReset();
    }

    // normal reset
    if (rcv == 'x') {
      NVIC_SystemReset();
    }

    // enable CDP mode
    if (rcv == 'C') {
      puts("switching USB to CDP mode\n");
      set_usb_power_mode(USB_POWER_CDP);
    }
    if (rcv == 'c') {
      puts("switching USB to client mode\n");
      set_usb_power_mode(USB_POWER_CLIENT);
    }
    if (rcv == 'D') {
      puts("switching USB to DCP mode\n");
      set_usb_power_mode(USB_POWER_DCP);
    }
  }
}

// ***************************** USB port *****************************

int get_health_pkt(void *dat) {
  struct __attribute__((packed)) {
    uint32_t voltage;
    uint32_t current;
    uint8_t started;
    uint8_t controls_allowed;
    uint8_t gas_interceptor_detected;
    uint8_t started_signal_detected;
    uint8_t started_alt;
  } *health = dat;

  //Voltage will be measured in mv. 5000 = 5V
  uint32_t voltage = adc_get(ADCCHAN_VOLTAGE);
  if (revision == PANDA_REV_AB) {
    //REVB has a 100, 27 (27/127) voltage divider
    //Here is the calculation for the scale
    //ADCV = VIN_S * (27/127) * (4095/3.3)
    //RETVAL = ADCV * s = VIN_S*1000
    //s = 1000/((4095/3.3)*(27/127)) = 3.79053046

    //Avoid needing floating point math
    health->voltage = (voltage * 3791) / 1000;
  } else {
    //REVC has a 10, 1 (1/11) voltage divider
    //Here is the calculation for the scale (s)
    //ADCV = VIN_S * (1/11) * (4095/3.3)
    //RETVAL = ADCV * s = VIN_S*1000
    //s = 1000/((4095/3.3)*(1/11)) = 8.8623046875

    //Avoid needing floating point math
    health->voltage = (voltage * 8862) / 1000;
  }

#ifdef PANDA
  health->current = adc_get(ADCCHAN_CURRENT);
  int safety_ignition = safety_ignition_hook();
  if (safety_ignition < 0) {
    //Use the GPIO pin to determine ignition
    health->started = (GPIOA->IDR & (1 << 1)) == 0;
  } else {
    //Current safety hooks want to determine ignition (ex: GM)
    health->started = safety_ignition;
  }
#else
  health->current = 0;
  health->started = (GPIOC->IDR & (1 << 13)) != 0;
#endif

  health->controls_allowed = controls_allowed;
  health->gas_interceptor_detected = gas_interceptor_detected;

  // DEPRECATED
  health->started_alt = 0;
  health->started_signal_detected = 0;

  return sizeof(*health);
}

int usb_cb_ep1_in(uint8_t *usbdata, int len, int hardwired) {
  CAN_FIFOMailBox_TypeDef *reply = (CAN_FIFOMailBox_TypeDef *)usbdata;
  int ilen = 0;
  while (ilen < min(len/0x10, 4) && can_pop(&can_rx_q, &reply[ilen])) ilen++;
  return ilen*0x10;
}

// send on serial, first byte to select the ring
void usb_cb_ep2_out(uint8_t *usbdata, int len, int hardwired) {
  if (len == 0) return;
  uart_ring *ur = get_ring_by_number(usbdata[0]);
  if (!ur) return;
  if ((usbdata[0] < 2) || safety_tx_lin_hook(usbdata[0]-2, usbdata+1, len-1)) {
    for (int i = 1; i < len; i++) while (!putc(ur, usbdata[i]));
  }
}

// send on CAN
void usb_cb_ep3_out(uint8_t *usbdata, int len, int hardwired) {
  int dpkt = 0;
  for (dpkt = 0; dpkt < len; dpkt += 0x10) {
    uint32_t *tf = (uint32_t*)(&usbdata[dpkt]);

    // make a copy
    CAN_FIFOMailBox_TypeDef to_push;
    to_push.RDHR = tf[3];
    to_push.RDLR = tf[2];
    to_push.RDTR = tf[1];
    to_push.RIR = tf[0];

    uint8_t bus_number = (to_push.RDTR >> 4) & CAN_BUS_NUM_MASK;
    can_send(&to_push, bus_number);
  }
}

int is_enumerated = 0;

void usb_cb_enumeration_complete() {
  puts("USB enumeration complete\n");
  is_enumerated = 1;
}

int usb_cb_control_msg(USB_Setup_TypeDef *setup, uint8_t *resp, int hardwired) {
  int resp_len = 0;
  uart_ring *ur = NULL;
  int i;
  switch (setup->b.bRequest) {
    // **** 0xc0: get CAN debug info
    case 0xc0:
      puts("can tx: "); puth(can_tx_cnt);
      puts(" txd: "); puth(can_txd_cnt);
      puts(" rx: "); puth(can_rx_cnt);
      puts(" err: "); puth(can_err_cnt);
      puts("\n");
      break;
    // **** 0xc1: is grey panda
    case 0xc1:
      resp[0] = is_grey_panda;
      resp_len = 1;
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
    // **** 0xd2: get health packet
    case 0xd2:
      resp_len = get_health_pkt(resp);
      break;
    // **** 0xd3: set fan speed
    case 0xd3:
      fan_set_speed(setup->b.wValue.w);
      break;
    // **** 0xd6: get version
    case 0xd6:
      COMPILE_TIME_ASSERT(sizeof(gitversion) <= MAX_RESP_LEN)
      memcpy(resp, gitversion, sizeof(gitversion));
      resp_len = sizeof(gitversion)-1;
      break;
    // **** 0xd8: reset ST
    case 0xd8:
      NVIC_SystemReset();
      break;
    // **** 0xd9: set ESP power
    case 0xd9:
      if (setup->b.wValue.w == 1) {
        set_esp_mode(ESP_ENABLED);
      } else if (setup->b.wValue.w == 2) {
        set_esp_mode(ESP_BOOTMODE);
      } else {
        set_esp_mode(ESP_DISABLED);
      }
      break;
    // **** 0xda: reset ESP, with optional boot mode
    case 0xda:
      set_esp_mode(ESP_DISABLED);
      delay(1000000);
      if (setup->b.wValue.w == 1) {
        set_esp_mode(ESP_BOOTMODE);
      } else {
        set_esp_mode(ESP_ENABLED);
      }
      delay(1000000);
      set_esp_mode(ESP_ENABLED);
      break;
    // **** 0xdb: set GMLAN multiplexing mode
    case 0xdb:
      #ifdef PANDA
        if (setup->b.wValue.w == 1) {
          // GMLAN ON
          if (setup->b.wIndex.w == 1) {
            can_set_gmlan(1);
          } else if (setup->b.wIndex.w == 2) {
            // might be ignored on rev b panda
            can_set_gmlan(2);
          }
        } else {
          can_set_gmlan(-1);
        }
      #endif
      break;
    // **** 0xdc: set safety mode
    case 0xdc:
      // this is the only way to leave silent mode
      // and it's blocked over WiFi
      // Allow ELM security mode to be set over wifi.
      if (hardwired || setup->b.wValue.w == SAFETY_NOOUTPUT || setup->b.wValue.w == SAFETY_ELM327) {
        safety_set_mode(setup->b.wValue.w, (int16_t)setup->b.wIndex.w);
        switch (setup->b.wValue.w) {
          case SAFETY_NOOUTPUT:
            can_silent = ALL_CAN_SILENT;
            break;
          case SAFETY_ELM327:
            can_silent = ALL_CAN_BUT_MAIN_SILENT;
            can_autobaud_enabled[0] = false;
            break;
          default:
            can_silent = ALL_CAN_LIVE;
            can_autobaud_enabled[0] = false;
            can_autobaud_enabled[1] = false;
            #ifdef PANDA
              can_autobaud_enabled[2] = false;
            #endif
            break;
        }
        can_init_all();
      }
      break;
    // **** 0xdd: enable can forwarding
    case 0xdd:
      // wValue = Can Bus Num to forward from
      // wIndex = Can Bus Num to forward to
      if (setup->b.wValue.w < BUS_MAX && setup->b.wIndex.w < BUS_MAX &&
          setup->b.wValue.w != setup->b.wIndex.w) { // set forwarding
        can_set_forwarding(setup->b.wValue.w, setup->b.wIndex.w & CAN_BUS_NUM_MASK);
      } else if(setup->b.wValue.w < BUS_MAX && setup->b.wIndex.w == 0xFF){ //Clear Forwarding
        can_set_forwarding(setup->b.wValue.w, -1);
      }
      break;
    // **** 0xde: set can bitrate
    case 0xde:
      if (setup->b.wValue.w < BUS_MAX) {
        can_autobaud_enabled[setup->b.wValue.w] = false;
        can_speed[setup->b.wValue.w] = setup->b.wIndex.w;
        can_init(CAN_NUM_FROM_BUS_NUM(setup->b.wValue.w));
      }
      break;
    // **** 0xe0: uart read
    case 0xe0:
      ur = get_ring_by_number(setup->b.wValue.w);
      if (!ur) break;
      if (ur == &esp_ring) uart_dma_drain();
      // read
      while ((resp_len < min(setup->b.wLength.w, MAX_RESP_LEN)) &&
                         getc(ur, (char*)&resp[resp_len])) {
        ++resp_len;
      }
      break;
    // **** 0xe1: uart set baud rate
    case 0xe1:
      ur = get_ring_by_number(setup->b.wValue.w);
      if (!ur) break;
      uart_set_baud(ur->uart, setup->b.wIndex.w);
      break;
    // **** 0xe2: uart set parity
    case 0xe2:
      ur = get_ring_by_number(setup->b.wValue.w);
      if (!ur) break;
      switch (setup->b.wIndex.w) {
        case 0:
          // disable parity, 8-bit
          ur->uart->CR1 &= ~(USART_CR1_PCE | USART_CR1_M);
          break;
        case 1:
          // even parity, 9-bit
          ur->uart->CR1 &= ~USART_CR1_PS;
          ur->uart->CR1 |= USART_CR1_PCE | USART_CR1_M;
          break;
        case 2:
          // odd parity, 9-bit
          ur->uart->CR1 |= USART_CR1_PS;
          ur->uart->CR1 |= USART_CR1_PCE | USART_CR1_M;
          break;
        default:
          break;
      }
      break;
    // **** 0xe4: uart set baud rate extended
    case 0xe4:
      ur = get_ring_by_number(setup->b.wValue.w);
      if (!ur) break;
      uart_set_baud(ur->uart, (int)setup->b.wIndex.w*300);
      break;
    // **** 0xe5: set CAN loopback (for testing)
    case 0xe5:
      can_loopback = (setup->b.wValue.w > 0);
      can_init_all();
      break;
    // **** 0xe6: set USB power
    case 0xe6:
      if (revision == PANDA_REV_C) {
        if (setup->b.wValue.w == 1) {
          puts("user setting CDP mode\n");
          set_usb_power_mode(USB_POWER_CDP);
        } else if (setup->b.wValue.w == 2) {
          puts("user setting DCP mode\n");
          set_usb_power_mode(USB_POWER_DCP);
        } else {
          puts("user setting CLIENT mode\n");
          set_usb_power_mode(USB_POWER_CLIENT);
        }
      }
      break;
    // **** 0xf0: do k-line wValue pulse on uart2 for Acura
    case 0xf0:
      if (setup->b.wValue.w == 1) {
        GPIOC->ODR &= ~(1 << 10);
        GPIOC->MODER &= ~GPIO_MODER_MODER10_1;
        GPIOC->MODER |= GPIO_MODER_MODER10_0;
      } else {
        GPIOC->ODR &= ~(1 << 12);
        GPIOC->MODER &= ~GPIO_MODER_MODER12_1;
        GPIOC->MODER |= GPIO_MODER_MODER12_0;
      }

      for (i = 0; i < 80; i++) {
        delay(8000);
        if (setup->b.wValue.w == 1) {
          GPIOC->ODR |= (1 << 10);
          GPIOC->ODR &= ~(1 << 10);
        } else {
          GPIOC->ODR |= (1 << 12);
          GPIOC->ODR &= ~(1 << 12);
        }
      }

      if (setup->b.wValue.w == 1) {
        GPIOC->MODER &= ~GPIO_MODER_MODER10_0;
        GPIOC->MODER |= GPIO_MODER_MODER10_1;
      } else {
        GPIOC->MODER &= ~GPIO_MODER_MODER12_0;
        GPIOC->MODER |= GPIO_MODER_MODER12_1;
      }

      delay(140 * 9000);
      break;
    // **** 0xf1: Clear CAN ring buffer.
    case 0xf1:
      if (setup->b.wValue.w == 0xFFFF) {
        puts("Clearing CAN Rx queue\n");
        can_clear(&can_rx_q);
      } else if (setup->b.wValue.w < BUS_MAX) {
        puts("Clearing CAN Tx queue\n");
        can_clear(can_queues[setup->b.wValue.w]);
      }
      break;
    // **** 0xf2: Clear UART ring buffer.
    case 0xf2:
      {
        uart_ring * rb = get_ring_by_number(setup->b.wValue.w);
        if (rb) {
          puts("Clearing UART queue.\n");
          clear_uart_buff(rb);
        }
        break;
      }
    default:
      puts("NO HANDLER ");
      puth(setup->b.bRequest);
      puts("\n");
      break;
  }
  return resp_len;
}

#ifdef PANDA
int spi_cb_rx(uint8_t *data, int len, uint8_t *data_out) {
  // data[0]  = endpoint
  // data[2]  = length
  // data[4:] = data

  int resp_len = 0;
  switch (data[0]) {
    case 0:
      // control transfer
      resp_len = usb_cb_control_msg((USB_Setup_TypeDef *)(data+4), data_out, 0);
      break;
    case 1:
      // ep 1, read
      resp_len = usb_cb_ep1_in(data_out, 0x40, 0);
      break;
    case 2:
      // ep 2, send serial
      usb_cb_ep2_out(data+4, data[2], 0);
      break;
    case 3:
      // ep 3, send CAN
      usb_cb_ep3_out(data+4, data[2], 0);
      break;
  }
  return resp_len;
}

#else

int spi_cb_rx(uint8_t *data, int len, uint8_t *data_out) { return 0; };

#endif


// ***************************** main code *****************************

void __initialize_hardware_early() {
  early();
}

void __attribute__ ((noinline)) enable_fpu() {
  // enable the FPU
  SCB->CPACR |= ((3UL << 10*2) | (3UL << 11*2));
}

int main() {
  // shouldn't have interrupts here, but just in case
  __disable_irq();

  // init early devices
  clock_init();
  periph_init();
  detect();

  // print hello
  puts("\n\n\n************************ MAIN START ************************\n");

  // detect the revision and init the GPIOs
  puts("config:\n");
  #ifdef PANDA
    puts(revision == PANDA_REV_C ? "  panda rev c\n" : "  panda rev a or b\n");
  #else
    puts("  legacy\n");
  #endif
  puts(has_external_debug_serial ? "  real serial\n" : "  USB serial\n");
  puts(is_giant_panda ? "  GIANTpanda detected\n" : "  not GIANTpanda\n");
  puts(is_grey_panda ? "  gray panda detected!\n" : "  white panda\n");
  puts(is_entering_bootmode ? "  ESP wants bootmode\n" : "  no bootmode\n");
  gpio_init();

#ifdef PANDA
  // panda has an FPU, let's use it!
  enable_fpu();
#endif

  // enable main uart if it's connected
  if (has_external_debug_serial) {
    // WEIRDNESS: without this gate around the UART, it would "crash", but only if the ESP is enabled
    // assuming it's because the lines were left floating and spurious noise was on them
    uart_init(USART2, 115200);
  }

#ifdef PANDA
  if (is_grey_panda) {
    uart_init(USART1, 9600);
  } else {
    // enable ESP uart
    uart_init(USART1, 115200);
  }
  // enable LIN
  uart_init(UART5, 10400);
  UART5->CR2 |= USART_CR2_LINEN;
  uart_init(USART3, 10400);
  USART3->CR2 |= USART_CR2_LINEN;
#endif

  // init microsecond system timer
  // increments 1000000 times per second
  // generate an update to set the prescaler
  TIM2->PSC = 48-1;
  TIM2->CR1 = TIM_CR1_CEN;
  TIM2->EGR = TIM_EGR_UG;
  // use TIM2->CNT to read

  // enable USB
  usb_init();

  // default to silent mode to prevent issues with Ford
  safety_set_mode(SAFETY_NOOUTPUT, 0);
  can_silent = ALL_CAN_SILENT;
  can_init_all();

  adc_init();

#ifdef PANDA
  spi_init();
#endif

  // set PWM
  fan_init();
  fan_set_speed(0);

  puts("**** INTERRUPTS ON ****\n");

  __enable_irq();

  // if the error interrupt is enabled to quickly when the CAN bus is active
  // something bad happens and you can't connect to the device over USB
  delay(10000000);
  CAN1->IER |= CAN_IER_ERRIE | CAN_IER_LECIE;

  // LED should keep on blinking all the time
  uint64_t cnt = 0;

  #ifdef PANDA
    uint64_t marker = 0;
    #define CURRENT_THRESHOLD 0xF00
    #define CLICKS 8
  #endif

  for (cnt=0;;cnt++) {
    can_live = pending_can_live;

    //puth(usart1_dma); puts(" "); puth(DMA2_Stream5->M0AR); puts(" "); puth(DMA2_Stream5->NDTR); puts("\n");

    #ifdef PANDA
      int current = adc_get(ADCCHAN_CURRENT);

      switch (usb_power_mode) {
        case USB_POWER_CLIENT:
          if ((cnt-marker) >= CLICKS) {
            if (!is_enumerated) {
              puts("USBP: didn't enumerate, switching to CDP mode\n");
              // switch to CDP
              set_usb_power_mode(USB_POWER_CDP);
              marker = cnt;
            }
          }
          // keep resetting the timer if it's enumerated
          if (is_enumerated) {
            marker = cnt;
          }
          break;
        case USB_POWER_CDP:
          // been CLICKS clicks since we switched to CDP
          if ((cnt-marker) >= CLICKS) {
            // measure current draw, if positive and no enumeration, switch to DCP
            if (!is_enumerated && current < CURRENT_THRESHOLD) {
              puts("USBP: no enumeration with current draw, switching to DCP mode\n");
              set_usb_power_mode(USB_POWER_DCP);
              marker = cnt;
            }
          }
          // keep resetting the timer if there's no current draw in CDP
          if (current >= CURRENT_THRESHOLD) {
            marker = cnt;
          }
          break;
        case USB_POWER_DCP:
          // been at least CLICKS clicks since we switched to DCP
          if ((cnt-marker) >= CLICKS) {
            // if no current draw, switch back to CDP
            if (current >= CURRENT_THRESHOLD) {
              puts("USBP: no current draw, switching back to CDP mode\n");
              set_usb_power_mode(USB_POWER_CDP);
              marker = cnt;
            }
          }
          // keep resetting the timer if there's current draw in DCP
          if (current < CURRENT_THRESHOLD) {
            marker = cnt;
          }
          break;
      }

      // ~0x9a = 500 ma
      /*puth(current);
      puts("\n");*/
    #endif

    // reset this every 16th pass
    if ((cnt&0xF) == 0) pending_can_live = 0;

    #ifdef DEBUG
      puts("** blink ");
      puth(can_rx_q.r_ptr); puts(" "); puth(can_rx_q.w_ptr); puts("  ");
      puth(can_tx1_q.r_ptr); puts(" "); puth(can_tx1_q.w_ptr); puts("  ");
      puth(can_tx2_q.r_ptr); puts(" "); puth(can_tx2_q.w_ptr); puts("\n");
    #endif

    // set green LED to be controls allowed
    set_led(LED_GREEN, controls_allowed);

    // blink the red LED
    int div_mode = ((usb_power_mode == USB_POWER_DCP) ? 4 : 1);

    for (int div_mode_loop = 0; div_mode_loop < div_mode; div_mode_loop++) {
      for (int fade = 0; fade < 1024; fade += 8) {
        for (int i = 0; i < 128/div_mode; i++) {
          set_led(LED_RED, 0);
          if (fade < 512) { delay(512-fade); } else { delay(fade-512); }
          set_led(LED_RED, 1);
          if (fade < 512) { delay(fade); } else { delay(1024-fade); }
        }
      }
    }

    // turn off the blue LED, turned on by CAN
    #ifdef PANDA
      set_led(LED_BLUE, 0);
    #endif
  }

  return 0;
}
