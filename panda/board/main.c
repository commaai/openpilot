// ********************* Includes *********************
#include "board/config.h"

#include "board/drivers/led.h"
#include "board/drivers/pwm.h"
#include "board/drivers/usb.h"
#include "board/drivers/simple_watchdog.h"
#include "board/drivers/bootkick.h"

#include "board/early_init.h"
#include "board/provision.h"

#include "opendbc/safety/safety.h"

#include "board/health.h"

#include "board/drivers/can_common.h"

#include "board/drivers/fdcan.h"

#include "board/sys/power_saving.h"

#include "board/obj/gitversion.h"

#include "board/can_comms.h"
#include "board/main_comms.h"
#include "board/drivers/self_test.h"

// ****************************** safety mode ******************************

// this is the only way to leave silent mode
void set_safety_mode(uint16_t mode, uint16_t param) {
  uint16_t mode_copy = mode;
  if ((faults & PERMANENT_FAULTS) != 0U) {
    mode_copy = SAFETY_SILENT;
  }
  mpu_safety_open();
  int err = set_safety_hooks(mode_copy, param);
  if (err == -1) {
    print("Error: safety set mode failed. Falling back to SILENT\n");
    mode_copy = SAFETY_SILENT;
    err = set_safety_hooks(mode_copy, 0U);
    // TERMINAL ERROR: we can't continue if SILENT safety mode isn't succesfully set
    assert_fatal(err == 0, "Error: Failed setting SILENT mode. Hanging\n");
  }
  mpu_safety_close();
  safety_tx_blocked = 0;
  safety_rx_invalid = 0;

  switch (mode_copy) {
    case SAFETY_SILENT:
      set_intercept_relay(false, false);
      current_board->set_can_mode(CAN_MODE_NORMAL);
      can_silent = true;
      break;
    case SAFETY_NOOUTPUT:
      set_intercept_relay(false, false);
      current_board->set_can_mode(CAN_MODE_NORMAL);
      can_silent = false;
      break;
    case SAFETY_ELM327:
      set_intercept_relay(false, false);
      heartbeat_counter = 0U;
      heartbeat_lost = false;

      // Clear any pending messages in the can core (i.e. sending while comma power is unplugged)
      // TODO: rewrite using hardware queues rather than fifo to cancel specific messages
      can_clear_send(CANIF_FROM_CAN_NUM(1), 1);
      if (param == 0U) {
        current_board->set_can_mode(CAN_MODE_OBD_CAN2);
      } else {
        current_board->set_can_mode(CAN_MODE_NORMAL);
      }
      can_silent = false;
      break;
    default:
      set_intercept_relay(true, false);
      heartbeat_counter = 0U;
      heartbeat_lost = false;
      current_board->set_can_mode(CAN_MODE_NORMAL);
      can_silent = false;
      break;
  }
  can_init_all();
}

// SILENT clears controls_allowed and opens the relay
void fault_safe_state(void) {
  set_safety_mode(SAFETY_SILENT, 0U);
}

bool is_car_safety_mode(uint16_t mode) {
  return (mode != SAFETY_SILENT) &&
         (mode != SAFETY_NOOUTPUT) &&
         (mode != SAFETY_ALLOUTPUT) &&
         (mode != SAFETY_ELM327);
}

// ***************************** main code *****************************

// cppcheck-suppress unusedFunction ; used in headers not included in cppcheck
// cppcheck-suppress misra-c2012-8.4
void __initialize_hardware_early(void) {
  early_initialization();
}

static void __attribute__ ((noinline)) enable_fpu(void) {
  // enable the FPU
  SCB->CPACR |= ((3UL << (10U * 2U)) | (3UL << (11U * 2U)));
}

// go into SILENT when heartbeat isn't received for this amount of seconds.
#define HEARTBEAT_IGNITION_CNT_ON 5U
#define HEARTBEAT_IGNITION_CNT_OFF 2U

// called at 8Hz
static void tick_handler(void) {
  if ((TICK_TIMER->SR & TIM_SR_UIF) != 0U) {
    // cleared first, a flag cleared on the way out can retrigger the vector
    TICK_TIMER->SR = 0;
    static uint32_t siren_countdown = 0; // siren plays while countdown > 0
    static uint32_t controls_allowed_countdown = 0;
    static uint8_t prev_harness_status = HARNESS_STATUS_NC;
    static uint8_t loop_counter = 0U;
    static bool relay_malfunction_prev = false;

    heartbeat_tick_counter += 1U;

    // start the siren early on power loss
    if ((controls_allowed_countdown > 0U) && (heartbeat_tick_counter > 2U) && (current_board->read_voltage_mV() < 6000U)) {
      siren_countdown = 3U;
    }

    // siren
    current_board->set_siren((loop_counter & 1U) && (siren_enabled || (siren_countdown > 0U)));

    // tick drivers at 8Hz
    fan_tick();
    harness_tick();
    simple_watchdog_kick();
    sound_tick();

    if (relay_malfunction_prev != relay_malfunction) {
      if (relay_malfunction) {
        fault_occurred(FAULT_RELAY_MALFUNCTION);
      } else {
        fault_recovered(FAULT_RELAY_MALFUNCTION);
      }
    }
    relay_malfunction_prev = relay_malfunction;

    // re-init everything that uses harness status
    if (harness.status != prev_harness_status) {
      prev_harness_status = harness.status;
      can_set_orientation(harness.status == HARNESS_STATUS_FLIPPED);

      // re-init everything that uses harness status
      can_init_all();
      set_safety_mode(current_safety_mode, current_safety_param);
      set_power_save_state(power_save_enabled);
    }

    // decimated to 1Hz
    if (loop_counter == 0U) {
      static timer_check_t tick_timer_check = {0};
      uint32_t tick_dt = 0U;
      if (timer_check(&tick_timer_check, &tick_dt)) {
        fault_occurred_value(FAULT_TIMER, tick_dt);
      }
      // set green LED to be controls allowed
      led_set(LED_GREEN, controls_allowed);

      // turn off the blue LED, turned on by CAN
      // unless we are in power saving mode
      led_set(LED_BLUE, (uptime_cnt & 1U) && power_save_enabled);

      const bool recent_heartbeat = heartbeat_counter == 0U;

      // tick drivers at 1Hz
      bool started = harness_check_ignition() || ignition_can;
      bootkick_tick(started, recent_heartbeat);

      // increase heartbeat counter and cap it at the uint32 limit
      if (heartbeat_counter < UINT32_MAX) {
        heartbeat_counter += 1U;
      }

      // disabling heartbeat not allowed while in safety mode
      if (is_car_safety_mode(current_safety_mode)) {
        heartbeat_disabled = false;
      }

      if (siren_countdown > 0U) {
        siren_countdown -= 1U;
      }

      if (controls_allowed || heartbeat_engaged) {
        controls_allowed_countdown = 5U;
      } else if (controls_allowed_countdown > 0U) {
        controls_allowed_countdown -= 1U;
      } else {

      }

      // exit controls allowed if unused by openpilot for a few seconds
      if (controls_allowed && !heartbeat_engaged) {
        heartbeat_engaged_mismatches += 1U;
        if (heartbeat_engaged_mismatches >= 3U) {
          mpu_safety_open();
          controls_allowed = false;
          mpu_safety_close();
        }
      } else {
        heartbeat_engaged_mismatches = 0U;
      }

      if (!heartbeat_disabled) {
        // if the heartbeat has been gone for a while, go to SILENT safety mode and enter power save
        if (heartbeat_counter >= (started ? HEARTBEAT_IGNITION_CNT_ON : HEARTBEAT_IGNITION_CNT_OFF)) {
          print("device hasn't sent a heartbeat for 0x");
          puth(heartbeat_counter);
          print(" seconds. Safety is set to SILENT mode.\n");

          if (controls_allowed_countdown > 0U) {
            siren_countdown = 3U;
            controls_allowed_countdown = 0U;
          }

          // set flag to indicate the heartbeat was lost
          if (is_car_safety_mode(current_safety_mode)) {
            heartbeat_lost = true;
          }

          // clear heartbeat engaged state
          heartbeat_engaged = false;

          if (current_safety_mode != SAFETY_SILENT) {
            set_safety_mode(SAFETY_SILENT, 0U);
          }

          if (!power_save_enabled) {
            set_power_save_state(true);
          }

          // Also disable IR when the heartbeat goes missing
          current_board->set_ir_power(0U);

          // Run fan when device is up but not talking to us.
          // The bootloader enables the SOM GPIO on boot.
          fan_set_power(current_board->read_som_gpio() ? 30U : 0U);
        }
      }

      // check registers
      check_registers();
      clock_check();
      interrupt_timer_check();
      nvic_check();
      check_option_bytes();

      // set ignition_can to false after 2s of no CAN seen
      if (ignition_can_cnt > 2U) {
        ignition_can = false;
      }

      // on to the next one
      uptime_cnt += 1U;
      safety_mode_cnt += 1U;
      ignition_can_cnt += 1U;

      // synchronous safety check
      mpu_safety_open();
      safety_tick();
      mpu_safety_close();
      mpu_check();
    }

    self_test_tick();

    loop_counter++;
    loop_counter %= 8U;
  } else {
    // TIM12 shares the vector with the TIM8 break, which has no interrupts enabled
    fault_occurred_value(FAULT_UNUSED_INTERRUPT_HANDLED, __get_IPSR());
  }
}

int main(void) {
  // Init interrupt table
  init_interrupts(true);

  // register_map is zeroed with .bss after early_initialization
  register_set(&(SCB->VTOR), (uint32_t)&g_pfnVectors, SCB_VTOR_TBLOFF_Msk);

  // shouldn't have interrupts here, but just in case
  disable_interrupts();

  // init early devices
  clock_init();
  peripherals_init();
  clock_check_init();
  detect_board_type();
  led_init();
  // red+green leds enabled until succesful USB/SPI init, as a debug indicator
  led_set(LED_RED, true);
  led_set(LED_GREEN, true);
  adc_init(ADC1);
  dts_init();

  // print hello
  print("\n\n\n************************ MAIN START ************************\n");

  // check for non-supported board types
  assert_fatal(hw_type != HW_TYPE_UNKNOWN, "Unsupported board type");

  print("Config:\n");
  print("  Board type: 0x"); puth(hw_type); print("\n");
  print("  MCU UID: "); hexdump((const uint8_t *)UID_BASE, 12);

  // init board
  current_board->init();
  current_board->set_can_mode(CAN_MODE_NORMAL);
  harness_init();

  // panda has an FPU, let's use it!
  enable_fpu();

  microsecond_timer_init();

  current_board->set_siren(false);
  if (current_board->has_fan) {
    fan_init();
  }

  mpu_init();

  // init to SILENT and can silent
  set_safety_mode(SAFETY_SILENT, 0U);

  // enable CAN TXs
  enable_can_transceivers(true);

  // init watchdog for heartbeat loop, fed at 8Hz
  simple_watchdog_init(FAULT_HEARTBEAT_LOOP_WATCHDOG, (3U * 1000000U / 8U));

  // 8Hz timer
  REGISTER_INTERRUPT(TICK_TIMER_IRQ, tick_handler, 10U, FAULT_INTERRUPT_RATE_TICK)
  interrupts[TICK_TIMER_IRQ].min_call_rate = 4U;
  tick_timer_init();

  self_test_init();

  // enable USB (right before interrupts or enum can fail!)
  usb_init();

  if (current_board->has_spi) {
    gpio_spi_init();
    spi_init();
  }

  led_set(LED_RED, false);
  led_set(LED_GREEN, false);
  led_set(LED_BLUE, false);

  if (fault_cookie == NMI_COOKIE) {
    reset_flags |= HEALTH_FLAG_NMI_RESET;
    print("NMI PC 0x"); puth(fault_pc); print("\n");
  }
  if (fault_cookie == HARDFAULT_COOKIE) {
    reset_flags |= HEALTH_FLAG_HARDFAULT_RESET;
    print("HardFault CFSR 0x"); puth(fault_cfsr); print(" address 0x"); puth(fault_addr); print(" PC 0x"); puth(fault_pc); print("\n");
  }
  fault_cookie = 0U;
  // cleared so the next boot shows only its own reset
  print("RSR 0x"); puth(RCC->RSR); print("\n");
  RCC->RSR |= RCC_RSR_RMVF;

  print("**** INTERRUPTS ON ****\n");
  enable_interrupts();

  // LED should keep on blinking all the time
  while (true) {
    #ifdef ALLOW_DEBUG
    if (stop_mode_requested) {
      enter_stop_mode();
    }
    #endif
    if (!power_save_enabled) {
      // ADC_SM_3, takes milliseconds so not in an interrupt
      adc_check(current_board->avdd_mV);

      // useful for debugging, fade breaks = panda is overloaded
      for (uint32_t fade = 0U; fade < MAX_LED_FADE; fade += 1U) {
        led_set(LED_RED, true);
        delay(fade >> 4);
        led_set(LED_RED, false);
        delay((MAX_LED_FADE - fade) >> 4);
      }

      for (uint32_t fade = MAX_LED_FADE; fade > 0U; fade -= 1U) {
        led_set(LED_RED, true);
        delay(fade >> 4);
        led_set(LED_RED, false);
        delay((MAX_LED_FADE - fade) >> 4);
      }
    } else {
      if ((hw_type == HW_TYPE_CUATRO) && !current_board->read_som_gpio()) {
        assert_fatal(current_safety_mode == SAFETY_SILENT, "Error: Entering low power mode while not in SAFETY_SILENT. Hanging\n");
        enter_stop_mode(); // deep sleep, wakes on CAN or SBU activity
        assert_fatal(false, "Error: enter_stop_mode returned after system reset. Hanging\n");
      }
      // cppcheck-suppress misra-c2012-17.3 ; CMSIS __WFI macro expands to inline asm
      __WFI();
      SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
    }
  }

  return 0;
}
