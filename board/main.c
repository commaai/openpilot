//#define DEBUG
//#define CAN_LOOPBACK_MODE
//#define USE_INTERNAL_OSC
//#define OLD_BOARD
//#define ENABLE_CURRENT_SENSOR
//#define ENABLE_SPI

#define USB_VID 0xbbaa
#define USB_PID 0xddcc

#define NULL ((void*)0)

// *** end config ***

#ifdef STM32F4
  #define PANDA
  #include "stm32f4xx.h"
#else
  #include "stm32f2xx.h"
#endif

#include "obj/gitversion.h"

#define ENTER_BOOTLOADER_MAGIC 0xdeadbeef
uint32_t enter_bootloader_mode;


// debug safety check: is controls allowed?
int controls_allowed = 0;
int started = 0;
int can_live = 0, pending_can_live = 0;

// optional features
int gas_interceptor_detected = 0;
int started_signal_detected = 0;

// detect high on UART
// TODO: check for UART high
int has_external_debug_serial = 1;

// ********************* instantiate queues *********************

#define FIFO_SIZE 0x100

typedef struct {
  uint8_t w_ptr;
  uint8_t r_ptr;
  CAN_FIFOMailBox_TypeDef elems[FIFO_SIZE];
} can_ring;

can_ring can_rx_q = { .w_ptr = 0, .r_ptr = 0 };
can_ring can_tx1_q = { .w_ptr = 0, .r_ptr = 0 };
can_ring can_tx2_q = { .w_ptr = 0, .r_ptr = 0 };
can_ring can_tx3_q = { .w_ptr = 0, .r_ptr = 0 };

// ********************* interrupt safe queue *********************

inline int pop(can_ring *q, CAN_FIFOMailBox_TypeDef *elem) {
  if (q->w_ptr != q->r_ptr) {
    *elem = q->elems[q->r_ptr];
    q->r_ptr += 1;
    return 1;
  }
  return 0;
}

inline int push(can_ring *q, CAN_FIFOMailBox_TypeDef *elem) {
  uint8_t next_w_ptr = q->w_ptr + 1;
  if (next_w_ptr != q->r_ptr) {
    q->elems[q->w_ptr] = *elem;
    q->w_ptr = next_w_ptr;
    return 1;
  }
  return 0;
}

// ***************************** serial port queues *****************************

typedef struct uart_ring {
  uint8_t w_ptr_tx;
  uint8_t r_ptr_tx;
  uint8_t elems_tx[FIFO_SIZE];
  uint8_t w_ptr_rx;
  uint8_t r_ptr_rx;
  uint8_t elems_rx[FIFO_SIZE];
  USART_TypeDef *uart;
  void (*callback)(struct uart_ring*);
} uart_ring;

inline int putc(uart_ring *q, char elem);

// esp = USART1
uart_ring esp_ring = { .w_ptr_tx = 0, .r_ptr_tx = 0,
                       .w_ptr_rx = 0, .r_ptr_rx = 0,
                       .uart = USART1 };

// lin1, K-LINE = UART5
// lin2, L-LINE = USART3
uart_ring lin1_ring = { .w_ptr_tx = 0, .r_ptr_tx = 0,
                        .w_ptr_rx = 0, .r_ptr_rx = 0,
                        .uart = UART5 };
uart_ring lin2_ring = { .w_ptr_tx = 0, .r_ptr_tx = 0,
                        .w_ptr_rx = 0, .r_ptr_rx = 0,
                        .uart = USART3 };

// debug = USART2
void debug_ring_callback(uart_ring *ring);
uart_ring debug_ring = { .w_ptr_tx = 0, .r_ptr_tx = 0,
                         .w_ptr_rx = 0, .r_ptr_rx = 0,
                         .uart = USART2,
                         .callback = debug_ring_callback};


uart_ring *get_ring_by_number(int a) {
  switch(a) {
    case 0:
      return &debug_ring;
    case 1:
      return &esp_ring;
    case 2:
      return &lin1_ring;
    case 3:
      return &lin2_ring;
    default:
      return NULL;
  }
}


void debug_ring_callback(uart_ring *ring) {
  char rcv;
  while (getc(ring, &rcv)) {
    putc(ring, rcv);

    // jump to DFU flash
    if (rcv == 'z') {
      enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
      NVIC_SystemReset();
    }
  }
}

// ***************************** serial port *****************************

void uart_ring_process(uart_ring *q) {
  // TODO: check if external serial is connected
  int sr = q->uart->SR;

  if (q->w_ptr_tx != q->r_ptr_tx) {
    if (sr & USART_SR_TXE) {
      q->uart->DR = q->elems_tx[q->r_ptr_tx];
      q->r_ptr_tx += 1;
    } else {
      // push on interrupt later
      q->uart->CR1 |= USART_CR1_TXEIE;
    }
  } else {
    // nothing to send
    q->uart->CR1 &= ~USART_CR1_TXEIE;
  }

  if (sr & USART_SR_RXNE) {
    uint8_t c = q->uart->DR;  // TODO: can drop packets
    uint8_t next_w_ptr = q->w_ptr_rx + 1;
    if (next_w_ptr != q->r_ptr_rx) {
      q->elems_rx[q->w_ptr_rx] = c;
      q->w_ptr_rx = next_w_ptr;
      if (q->callback) q->callback(q);
    }
  }
}

// interrupt boilerplate

void USART1_IRQHandler(void) {
  NVIC_DisableIRQ(USART1_IRQn);
  uart_ring_process(&esp_ring);
  NVIC_EnableIRQ(USART1_IRQn);
}

void USART2_IRQHandler(void) {
  NVIC_DisableIRQ(USART2_IRQn);
  uart_ring_process(&debug_ring);
  NVIC_EnableIRQ(USART2_IRQn);
}

void USART3_IRQHandler(void) {
  NVIC_DisableIRQ(USART3_IRQn);
  uart_ring_process(&lin2_ring);
  NVIC_EnableIRQ(USART3_IRQn);
}

void UART5_IRQHandler(void) {
  NVIC_DisableIRQ(UART5_IRQn);
  uart_ring_process(&lin1_ring);
  NVIC_EnableIRQ(UART5_IRQn);
}

inline int getc(uart_ring *q, char *elem) {
  if (q->w_ptr_rx != q->r_ptr_rx) {
    *elem = q->elems_rx[q->r_ptr_rx];
    q->r_ptr_rx += 1;
    return 1;
  }
  return 0;
}

inline int injectc(uart_ring *q, char elem) {
  uint8_t next_w_ptr = q->w_ptr_rx + 1;
  int ret = 0;
  if (next_w_ptr != q->r_ptr_rx) {
    q->elems_rx[q->w_ptr_rx] = elem;
    q->w_ptr_rx = next_w_ptr;
    ret = 1;
  }
  return ret;
}

inline int putc(uart_ring *q, char elem) {
  uint8_t next_w_ptr = q->w_ptr_tx + 1;
  int ret = 0;
  if (next_w_ptr != q->r_ptr_tx) {
    q->elems_tx[q->w_ptr_tx] = elem;
    q->w_ptr_tx = next_w_ptr;
    ret = 1;
  }
  uart_ring_process(q);
  return ret;
}

// ********************* includes *********************

#include "libc.h"
#include "adc.h"
#include "timer.h"
#include "usb.h"
#include "can.h"
#include "spi.h"

// ***************************** CAN *****************************

void process_can(CAN_TypeDef *CAN, can_ring *can_q, int can_number) {
  #ifdef DEBUG
    puts("process CAN TX\n");
  #endif

  // add successfully transmitted message to my fifo
  if ((CAN->TSR & CAN_TSR_TXOK0) == CAN_TSR_TXOK0) {
    CAN_FIFOMailBox_TypeDef to_push;
    to_push.RIR = CAN->sTxMailBox[0].TIR;
    to_push.RDTR = (CAN->sTxMailBox[0].TDTR & 0xFFFF000F) | ((can_number+2) << 4);
    to_push.RDLR = CAN->sTxMailBox[0].TDLR;
    to_push.RDHR = CAN->sTxMailBox[0].TDHR;
    push(&can_rx_q, &to_push);
  }

  // check for empty mailbox
  CAN_FIFOMailBox_TypeDef to_send;
  if ((CAN->TSR & CAN_TSR_TME0) == CAN_TSR_TME0) {
    if (pop(can_q, &to_send)) {

      // BRAKE: safety check
      if ((to_send.RIR>>21) == 0x1FA) {
        if (controls_allowed) {
          to_send.RDLR &= 0xFFFFFF3F;
        } else {
          to_send.RDLR &= 0xFFFF0000;
        }
      }

      // STEER: safety check
      if ((to_send.RIR>>21) == 0xE4) {
        if (controls_allowed) {
          to_send.RDLR &= 0xFFFFFFFF;
        } else {
          to_send.RDLR &= 0xFFFF0000;
        }
      } 

      // GAS: safety check
      if ((to_send.RIR>>21) == 0x200) {
        if (controls_allowed) {
          to_send.RDLR &= 0xFFFFFFFF;
        } else {
          to_send.RDLR &= 0xFFFF0000;
        }
      } 

      // only send if we have received a packet
      CAN->sTxMailBox[0].TDLR = to_send.RDLR;
      CAN->sTxMailBox[0].TDHR = to_send.RDHR;
      CAN->sTxMailBox[0].TDTR = to_send.RDTR;
      CAN->sTxMailBox[0].TIR = to_send.RIR;
    }
  }

  // clear interrupt
  CAN->TSR |= CAN_TSR_RQCP0;
}

// send more, possible for these to not trigger?
// CAN2 = 0   CAN1 = 1   CAN3 = 4
void CAN1_TX_IRQHandler() {
  process_can(CAN1, &can_tx1_q, 1);
}

void CAN2_TX_IRQHandler() {
  process_can(CAN2, &can_tx2_q, 0);
}

#ifdef CAN3
void CAN3_TX_IRQHandler() {
  process_can(CAN3, &can_tx3_q, 4);
}
#endif

// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button


// all commands: brake and steering
// if controls_allowed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

// CAN receive handlers
void can_rx(CAN_TypeDef *CAN, int can_number) {
  while (CAN->RF0R & CAN_RF0R_FMP0) {
    // can is live
    pending_can_live = 1;

    // add to my fifo
    CAN_FIFOMailBox_TypeDef to_push;
    to_push.RIR = CAN->sFIFOMailBox[0].RIR;
    // top 16-bits is the timestamp
    to_push.RDTR = (CAN->sFIFOMailBox[0].RDTR & 0xFFFF000F) | (can_number << 4);
    to_push.RDLR = CAN->sFIFOMailBox[0].RDLR;
    to_push.RDHR = CAN->sFIFOMailBox[0].RDHR;

    // state machine to enter and exit controls
    // 0x1A6 for the ILX, 0x296 for the Civic Touring
    if ((to_push.RIR>>21) == 0x1A6 || (to_push.RIR>>21) == 0x296) {
      int buttons = (to_push.RDLR & 0xE0) >> 5;
      if (buttons == 4 || buttons == 3) {
        controls_allowed = 1;
      } else if (buttons == 2) {
        controls_allowed = 0;
      }
    }

    // exit controls on brake press
    if ((to_push.RIR>>21) == 0x17C) {
      // bit 50
      if (to_push.RDHR & 0x200000) {
        controls_allowed = 0;
      }
    }

    // exit controls on gas press if interceptor
    if ((to_push.RIR>>21) == 0x201) {
      gas_interceptor_detected = 1;
      int gas = ((to_push.RDLR & 0xFF) << 8) | ((to_push.RDLR & 0xFF00) >> 8);
      if (gas > 328) {
        controls_allowed = 0;
      }
    }

    // exit controls on gas press if no interceptor
    if (!gas_interceptor_detected) {
      if ((to_push.RIR>>21) == 0x17C) {
        if (to_push.RDLR & 0xFF) {
          controls_allowed = 0;
        }
      }
    }

    push(&can_rx_q, &to_push);

    // next
    CAN->RF0R |= CAN_RF0R_RFOM0;
  }
}

void CAN1_RX0_IRQHandler() {
  //puts("CANRX1");
  //delay(10000);
  can_rx(CAN1, 1);
}

void CAN2_RX0_IRQHandler() {
  //puts("CANRX0");
  //delay(10000);
  can_rx(CAN2, 0);
}

#ifdef CAN3
void CAN3_RX0_IRQHandler() {
  //puts("CANRX0");
  //delay(10000);
  can_rx(CAN3, 4);
}
#endif

void CAN1_SCE_IRQHandler() {
  //puts("CAN1_SCE\n");
  can_sce(CAN1);
}

void CAN2_SCE_IRQHandler() {
  //puts("CAN2_SCE\n");
  can_sce(CAN2);
}

#ifdef CAN3
void CAN3_SCE_IRQHandler() {
  //puts("CAN3_SCE\n");
  can_sce(CAN3);
}
#endif


// ***************************** USB port *****************************

int get_health_pkt(void *dat) {
  struct {
    uint32_t voltage;
    uint32_t current;
    uint8_t started;
    uint8_t controls_allowed;
    uint8_t gas_interceptor_detected;
    uint8_t started_signal_detected;
  } *health = dat;
  health->voltage = adc_get(ADCCHAN_VOLTAGE);
#ifdef ENABLE_CURRENT_SENSOR
  health->current = adc_get(ADCCHAN_CURRENT);
#else
  health->current = 0;
#endif
  health->started = started;

  health->controls_allowed = controls_allowed;

  health->gas_interceptor_detected = gas_interceptor_detected;
  health->started_signal_detected = started_signal_detected;
  return sizeof(*health);
}

void set_fan_speed(int fan_speed) {
  #ifdef OLD_BOARD
    TIM3->CCR4 = fan_speed;
  #else
    TIM3->CCR3 = fan_speed;
  #endif
}

void usb_cb_ep1_in(int len) {
  CAN_FIFOMailBox_TypeDef reply[4];

  int ilen = 0;
  while (ilen < min(len/0x10, 4) && pop(&can_rx_q, &reply[ilen])) ilen++;

  #ifdef DEBUG
    puts("FIFO SENDING ");
    puth(ilen);
    puts("\n");
  #endif

  USB_WritePacket((void *)reply, ilen*0x10, 1);
}

// send on serial, first byte to select
void usb_cb_ep2_out(uint8_t *usbdata, int len) {
  int i;
  if (len == 0) return;
  uart_ring *ur = get_ring_by_number(usbdata[0]);
  if (!ur) return;
  for (i = 1; i < len; i++) while (!putc(ur, usbdata[i]));
}

// send on CAN
void usb_cb_ep3_out(uint8_t *usbdata, int len) {
  int dpkt = 0;
  for (dpkt = 0; dpkt < len; dpkt += 0x10) {
    uint32_t *tf = (uint32_t*)(&usbdata[dpkt]);
    
    int flags = (tf[1] >> 4) & 0xF;
    CAN_TypeDef *CAN;
    can_ring *can_q;
    int can_number = 0;
    if (flags == 1)  {
      CAN=CAN1;
      can_q = &can_tx1_q;
      can_number = 1;
    } else if (flags == 0) {
      CAN=CAN2;
      can_q = &can_tx2_q;
    #ifdef CAN3
    } else if (flags == 4) {
      CAN=CAN3;
      can_q = &can_tx3_q;
    #endif
    }

    // add CAN packet to send queue
    CAN_FIFOMailBox_TypeDef to_push;
    to_push.RDHR = tf[3];
    to_push.RDLR = tf[2];
    to_push.RDTR = tf[1] & 0xF;
    to_push.RIR = tf[0];
    push(can_q, &to_push);

    process_can(CAN, can_q, can_number);
  }
}


#define MAX_RESP_LEN 0x30
void usb_cb_control_msg() {
  uint8_t resp[MAX_RESP_LEN];
  int resp_len = 0;
  uart_ring *ur = NULL;
  switch (setup.b.bRequest) {
    case 0xd1:
      enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
      NVIC_SystemReset();
      break;
    case 0xd2:
      resp_len = get_health_pkt(resp);
      USB_WritePacket(resp, resp_len, 0);
      USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    case 0xd3:
      set_fan_speed(setup.b.wValue.w);
      USB_WritePacket(0, 0, 0);
      USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    case 0xd6: // GET_VERSION
      USB_WritePacket(gitversion, min(sizeof(gitversion), setup.b.wLength.w), 0);
      USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    case 0xd8: // RESET
      NVIC_SystemReset();
      break;
    case 0xda: // ESP RESET
      GPIOC->ODR &= ~(1 << 14);
      delay(1000000);
      GPIOC->ODR |= (1 << 14);
      USB_WritePacket(0, 0, 0);
      USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    case 0xe0: // uart read
      ur = get_ring_by_number(setup.b.wValue.w);
      if (!ur) break;
      if (setup.b.bRequest == 0xe0) {
        // read
        while (resp_len < min(setup.b.wLength.w, MAX_RESP_LEN) && getc(ur, &resp[resp_len])) {
          ++resp_len;
        }
        /*puts("uart read: ");
        puth(setup.b.wLength.w);
        puts(" ");
        puth(resp_len);
        puts("\n");*/
        USB_WritePacket(resp, resp_len, 0);
        USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      }
      break;
    default:
      puts("NO HANDLER ");
      puth(setup.b.bRequest);
      puts("\n");
      break;
  }
}


void OTG_FS_IRQHandler(void) {
  NVIC_DisableIRQ(OTG_FS_IRQn);
  //__disable_irq();
  usb_irqhandler();
  //__enable_irq();
  NVIC_EnableIRQ(OTG_FS_IRQn);
}

void ADC_IRQHandler(void) {
  puts("ADC_IRQ\n");
}

#ifdef ENABLE_SPI

#define SPI_BUF_SIZE 128
uint8_t spi_buf[SPI_BUF_SIZE];
int spi_buf_count = 0;
uint8_t spi_tx_buf[0x10];

void DMA2_Stream3_IRQHandler(void) {
  #ifdef DEBUG
    puts("DMA2\n");
  #endif
  DMA2->LIFCR = DMA_LIFCR_CTCIF3;

  pop(&can_rx_q, spi_tx_buf);
  spi_tx_dma(spi_tx_buf, 0x10);
}

void SPI1_IRQHandler(void) {
  // status is 0x43
  if (SPI1->SR & SPI_SR_RXNE) {
    uint8_t dat = SPI1->DR;
    /*spi_buf[spi_buf_count] = dat;
    if (spi_buf_count < SPI_BUF_SIZE-1) {
      spi_buf_count += 1;
    }*/
  }

  if (SPI1->SR & SPI_SR_TXE) {
    // all i send is U U U no matter what
    //SPI1->DR = 'U';
  }

  int stat = SPI1->SR;
  if (stat & ((~SPI_SR_RXNE) & (~SPI_SR_TXE) & (~SPI_SR_BSY))) {
    puts("SPI status: ");
    puth(stat);
    puts("\n");
  }
}

#endif

// ***************************** main code *****************************

void __initialize_hardware_early() {
  // if wrong chip, reboot
  volatile unsigned int id = DBGMCU->IDCODE;
  #ifdef STM32F4
    if ((id&0xFFF) != 0x463) enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
  #else
    if ((id&0xFFF) != 0x411) enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
  #endif

  // enable the ESP
  RCC->AHB1ENR = RCC_AHB1ENR_GPIOCEN;
  GPIOC->ODR = (1 << 14);
  GPIOC->MODER = GPIO_MODER_MODER14_0;

  if (enter_bootloader_mode == ENTER_BOOTLOADER_MAGIC) {
    enter_bootloader_mode = 0;
    void (*bootloader)(void) = (void (*)(void)) (*((uint32_t *)0x1fff0004));

    // jump to bootloader
    bootloader();

    // LOOP
    while(1);
  }
}

int main() {
  // init devices
  clock_init();

  gpio_init();

  // enable main uart
  uart_init(USART2, 115200);
  
  // enable ESP uart
  uart_init(USART1, 115200);

  // enable LIN
  uart_init(UART5, 10400);
  UART5->CR2 |= USART_CR2_LINEN;
  uart_init(USART3, 10400);
  USART3->CR2 |= USART_CR2_LINEN;

  usb_init();
  can_init(CAN1);
  can_init(CAN2);
#ifdef CAN3
  can_init(CAN3);
#endif
  adc_init();

#ifdef ENABLE_SPI
  spi_init();

  // set up DMA
  memset(spi_tx_buf, 0, 0x10);
  spi_tx_dma(spi_tx_buf, 0x10);
#endif

  // timer for fan PWM
  #ifdef OLD_BOARD
    TIM3->CCMR2 = TIM_CCMR2_OC4M_2 | TIM_CCMR2_OC4M_1;
    TIM3->CCER = TIM_CCER_CC4E;
  #else
    TIM3->CCMR2 = TIM_CCMR2_OC3M_2 | TIM_CCMR2_OC3M_1;
    TIM3->CCER = TIM_CCER_CC3E;
  #endif

  // max value of the timer
  // 64 makes it above the audible range
  //TIM3->ARR = 64;

  // 10 prescale makes it below the audible range
  timer_init(TIM3, 10);
  puth(DBGMCU->IDCODE);

  // set PWM
  set_fan_speed(65535);

  puts("**** INTERRUPTS ON ****\n");
  __disable_irq();

  // 4 uarts!
  NVIC_EnableIRQ(USART1_IRQn);
  NVIC_EnableIRQ(USART2_IRQn);
  NVIC_EnableIRQ(USART3_IRQn);
  NVIC_EnableIRQ(UART5_IRQn);

  NVIC_EnableIRQ(OTG_FS_IRQn);
  NVIC_EnableIRQ(ADC_IRQn);
  // CAN has so many interrupts!

  NVIC_EnableIRQ(CAN1_TX_IRQn);
  NVIC_EnableIRQ(CAN1_RX0_IRQn);
  NVIC_EnableIRQ(CAN1_SCE_IRQn);

  NVIC_EnableIRQ(CAN2_TX_IRQn);
  NVIC_EnableIRQ(CAN2_RX0_IRQn);
  NVIC_EnableIRQ(CAN2_SCE_IRQn);

#ifdef CAN3
  NVIC_EnableIRQ(CAN3_TX_IRQn);
  NVIC_EnableIRQ(CAN3_RX0_IRQn);
  NVIC_EnableIRQ(CAN3_SCE_IRQn);
#endif

#ifdef ENABLE_SPI
  NVIC_EnableIRQ(DMA2_Stream3_IRQn);
  NVIC_EnableIRQ(SPI1_IRQn);
#endif
  __enable_irq();

  // LED should keep on blinking all the time
  int cnt;
  for (cnt=0;;cnt++) {
    can_live = pending_can_live;

    // reset this every 16th pass
    if ((cnt&0xF) == 0) pending_can_live = 0;

    #ifdef DEBUG
      puts("** blink ");
      puth(can_rx_q.r_ptr); puts(" "); puth(can_rx_q.w_ptr); puts("  ");
      puth(can_tx1_q.r_ptr); puts(" "); puth(can_tx1_q.w_ptr); puts("  ");
      puth(can_tx2_q.r_ptr); puts(" "); puth(can_tx2_q.w_ptr); puts("\n");
    #endif

    /*puts("voltage: "); puth(adc_get(ADCCHAN_VOLTAGE)); puts("  ");
    puts("current: "); puth(adc_get(ADCCHAN_CURRENT)); puts("\n");*/

    // set LED to be controls allowed
    set_led(LED_GREEN, controls_allowed);

    // blink the red LED
    set_led(LED_RED, 0);
    delay(2000000);
    set_led(LED_RED, 1);
    delay(2000000);

    #ifdef ENABLE_SPI
      if (spi_buf_count > 0) {
        hexdump(spi_buf, spi_buf_count);
        spi_buf_count = 0;
      }
    #endif

    // started logic
    #ifdef PANDA
      int started_signal = (GPIOB->IDR & (1 << 12)) == 0;
    #else
      int started_signal = (GPIOC->IDR & (1 << 13)) != 0;
    #endif
    if (started_signal) { started_signal_detected = 1; }

    if (started_signal || (!started_signal_detected && can_live)) {
      started = 1;

      // turn on fan at half speed
      set_fan_speed(32768);
    } else {
      started = 0;

      // turn off fan
      set_fan_speed(0);
    }
  }

  return 0;
}

