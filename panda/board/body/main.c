#include <stdint.h>
#include <stdbool.h>

#include "board/config.h"
#include "board/drivers/led.h"
#include "board/drivers/pwm.h"
#include "board/drivers/usb.h"
#include "board/early_init.h"
#include "board/obj/gitversion.h"
#include "board/body/motor_control.h"
#include "board/body/can.h"
#include "opendbc/safety/safety.h"
#include "board/drivers/can_common.h"
#include "board/drivers/fdcan.h"
#include "board/can_comms.h"

extern int _app_start[0xc000];

#include "board/body/main_comms.h"

void debug_ring_callback(uart_ring *ring) {
  char rcv;
  while (get_char(ring, &rcv)) {
    (void)injectc(ring, rcv);
  }
}

void __attribute__ ((noinline)) enable_fpu(void) {
  SCB->CPACR |= ((3UL << (10U * 2U)) | (3UL << (11U * 2U)));
}

void __initialize_hardware_early(void) {
  enable_fpu();
  early_initialization();
}

volatile uint32_t tick_count = 0;

void tick_handler(void) {
  if (TICK_TIMER->SR != 0) {
    if (can_health[0].transmit_error_cnt >= 128) {
      (void)llcan_init(CANIF_FROM_CAN_NUM(0));
    }
    static bool led_on = false;
    led_set(LED_RED, led_on);
    led_on = !led_on;
    tick_count++;
  }
  TICK_TIMER->SR = 0;
}

int main(void) {
  disable_interrupts();
  init_interrupts(true);

  clock_init();
  peripherals_init();

  current_board = &board_body;
  hw_type = HW_TYPE_BODY;

  current_board->init();
  
  REGISTER_INTERRUPT(TICK_TIMER_IRQ, tick_handler, 10U, FAULT_INTERRUPT_RATE_TICK);

  led_init();
  microsecond_timer_init();
  tick_timer_init();
  usb_init();
  body_can_init();

  enable_interrupts();

  while (true) {
    uint32_t now = microsecond_timer_get();
    motor_speed_controller_update(now);
    body_can_periodic(now);
  }

  return 0;
}
