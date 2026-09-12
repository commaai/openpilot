#include <stdint.h>
#include <stdbool.h>

#include "board/config.h"
#include "board/drivers/led.h"
#include "board/drivers/pwm.h"
#include "board/drivers/usb.h"
#include "board/early_init.h"
#include "board/obj/gitversion.h"
#include "board/body/can.h"
#include "opendbc/safety/safety.h"
#include "board/drivers/can_common.h"
#include "board/drivers/fdcan.h"
#include "board/can_comms.h"
#include "board/body/dotstar.h"
#include "bldc/bldc.h"

extern int _app_start[0xc000];

#include "board/body/main_comms.h"

static volatile uint32_t tick_count = 0U;
static volatile uint32_t ignition_press_timestamp_us = 0U;
static volatile bool ignition = false;
static volatile bool plug_charging = false;

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

void bldc_tim8_handler(void) {
  if ((LEFT_TIM->SR & TIM_SR_UIF) != 0) {
    LEFT_TIM->SR = ~TIM_SR_UIF;
    bldc_step();
  }
}

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

static void exti15_10_handler(void) {
  if ((EXTI->PR1 & (1U << CHARGING_DETECT_PIN)) != 0U) {
    EXTI->PR1 = (1U << CHARGING_DETECT_PIN);
    plug_charging = (get_gpio_input(CHARGING_DETECT_PORT, CHARGING_DETECT_PIN) != 0);
  }

  if ((EXTI->PR1 & (1U << IGNITION_SW_PIN)) != 0U) {
    EXTI->PR1 = (1U << IGNITION_SW_PIN);

    static uint32_t last_press_event_us = 0U;
    const uint32_t debounce_us = 200000U; // 200 ms

    uint32_t now = microsecond_timer_get();
    bool pressed = (get_gpio_input(IGNITION_SW_PORT, IGNITION_SW_PIN) == 0);

    if (pressed) {
      if (get_ts_elapsed(now, last_press_event_us) > debounce_us) {
        last_press_event_us = now;
        ignition_press_timestamp_us = now;
        ignition = !ignition;
        set_gpio_output(OBDC_IGNITION_ON_PORT, OBDC_IGNITION_ON_PIN, ignition);
      }
    }
  }
}

int main(void) {
  disable_interrupts();
  init_interrupts(true);

  clock_init();
  peripherals_init();

  current_board = &board_body;
  hw_type = HW_TYPE_BODY;

  current_board->init();

  REGISTER_INTERRUPT(EXTI15_10_IRQn, exti15_10_handler, 10000U, FAULT_INTERRUPT_RATE_EXTI);
  NVIC_ClearPendingIRQ(EXTI15_10_IRQn);
  NVIC_EnableIRQ(EXTI15_10_IRQn);

  REGISTER_INTERRUPT(TIM8_UP_TIM13_IRQn, bldc_tim8_handler, 100000U, FAULT_INTERRUPT_RATE_TICK);
  NVIC_ClearPendingIRQ(TIM8_UP_TIM13_IRQn);
  NVIC_EnableIRQ(TIM8_UP_TIM13_IRQn);

  REGISTER_INTERRUPT(TICK_TIMER_IRQ, tick_handler, 10U, FAULT_INTERRUPT_RATE_TICK);

  led_init();
  microsecond_timer_init();
  tick_timer_init();
  usb_init();
  body_can_init();
  dotstar_init();
  bldc_init();
  enable_interrupts();

  plug_charging = (get_gpio_input(CHARGING_DETECT_PORT, CHARGING_DETECT_PIN) != 0);

  while (true) {
    uint32_t now = microsecond_timer_get();
    if (plug_charging) {
      motor_set_enable(false);
      dotstar_apply_breathe((dotstar_rgb_t){255U, 40U, 0U}, now, 2000000U);
    } else if (ignition) {
      dotstar_run_rainbow(now);
    } else {
      dotstar_apply_breathe((dotstar_rgb_t){0U, 255U, 10U}, now, 1500000U);
    }

    if (ignition) {
      motor_set_enable(true);
      body_can_periodic(now, ignition, plug_charging);
    } else {
      motor_set_enable(false);
    }

    dotstar_show();
  }

  return 0;
}