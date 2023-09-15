// ********************* Includes *********************
#include "config.h"

#include "drivers/pwm.h"
#include "drivers/usb.h"
#include "drivers/gmlan_alt.h"
#include "drivers/kline_init.h"
#include "drivers/simple_watchdog.h"
#include "drivers/logging.h"

#include "early_init.h"
#include "provision.h"

#include "safety.h"

#include "health.h"

#include "drivers/can_common.h"

#ifdef STM32H7
  #include "drivers/fdcan.h"
#else
  #include "drivers/bxcan.h"
#endif

#include "power_saving.h"

#include "obj/gitversion.h"

#include "can_comms.h"
#include "main_comms.h"


// ********************* Serial debugging *********************

bool check_started(void) {
  bool started = current_board->check_ignition() || ignition_can;
  ignition_seen |= started;
  return started;
}

void debug_ring_callback(uart_ring *ring) {
  char rcv;
  while (getc(ring, &rcv)) {
    (void)putc(ring, rcv);  // misra-c2012-17.7: cast to void is ok: debug function

    // only allow bootloader entry on debug builds
    #ifdef ALLOW_DEBUG
      // jump to DFU flash
      if (rcv == 'z') {
        enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
        NVIC_SystemReset();
      }
    #endif

    // normal reset
    if (rcv == 'x') {
      NVIC_SystemReset();
    }
  }
}

// ****************************** safety mode ******************************

// this is the only way to leave silent mode
void set_safety_mode(uint16_t mode, uint16_t param) {
  uint16_t mode_copy = mode;
  int err = set_safety_hooks(mode_copy, param);
  if (err == -1) {
    print("Error: safety set mode failed. Falling back to SILENT\n");
    mode_copy = SAFETY_SILENT;
    err = set_safety_hooks(mode_copy, 0U);
    if (err == -1) {
      print("Error: Failed setting SILENT mode. Hanging\n");
      while (true) {
        // TERMINAL ERROR: we can't continue if SILENT safety mode isn't succesfully set
      }
    }
  }
  safety_tx_blocked = 0;
  safety_rx_invalid = 0;

  switch (mode_copy) {
    case SAFETY_SILENT:
      set_intercept_relay(false, false);
      if (current_board->has_obd) {
        current_board->set_can_mode(CAN_MODE_NORMAL);
      }
      can_silent = ALL_CAN_SILENT;
      break;
    case SAFETY_NOOUTPUT:
      set_intercept_relay(false, false);
      if (current_board->has_obd) {
        current_board->set_can_mode(CAN_MODE_NORMAL);
      }
      can_silent = ALL_CAN_LIVE;
      break;
    case SAFETY_ELM327:
      set_intercept_relay(false, false);
      heartbeat_counter = 0U;
      heartbeat_lost = false;
      if (current_board->has_obd) {
        if (param == 0U) {
          current_board->set_can_mode(CAN_MODE_OBD_CAN2);
        } else {
          current_board->set_can_mode(CAN_MODE_NORMAL);
        }
      }
      can_silent = ALL_CAN_LIVE;
      break;
    default:
      set_intercept_relay(true, false);
      heartbeat_counter = 0U;
      heartbeat_lost = false;
      if (current_board->has_obd) {
        current_board->set_can_mode(CAN_MODE_NORMAL);
      }
      can_silent = ALL_CAN_LIVE;
      break;
  }
  can_init_all();
}

bool is_car_safety_mode(uint16_t mode) {
  return (mode != SAFETY_SILENT) &&
         (mode != SAFETY_NOOUTPUT) &&
         (mode != SAFETY_ALLOUTPUT) &&
         (mode != SAFETY_ELM327);
}

// ***************************** main code *****************************

// cppcheck-suppress unusedFunction ; used in headers not included in cppcheck
void __initialize_hardware_early(void) {
  early_initialization();
}

void __attribute__ ((noinline)) enable_fpu(void) {
  // enable the FPU
  SCB->CPACR |= ((3UL << (10U * 2U)) | (3UL << (11U * 2U)));
}

// go into SILENT when heartbeat isn't received for this amount of seconds.
#define HEARTBEAT_IGNITION_CNT_ON 5U
#define HEARTBEAT_IGNITION_CNT_OFF 2U

// called at 8Hz
uint8_t loop_counter = 0U;
uint8_t previous_harness_status = HARNESS_STATUS_NC;
uint32_t waiting_to_boot_count = 0;
bool waiting_to_boot = false;
void tick_handler(void) {
  if (TICK_TIMER->SR != 0) {
    // siren
    current_board->set_siren((loop_counter & 1U) && (siren_enabled || (siren_countdown > 0U)));

    // tick drivers at 8Hz
    fan_tick();
    usb_tick();
    harness_tick();
    simple_watchdog_kick();

    // decimated to 1Hz
    if (loop_counter == 0U) {
      can_live = pending_can_live;

      //puth(usart1_dma); print(" "); puth(DMA2_Stream5->M0AR); print(" "); puth(DMA2_Stream5->NDTR); print("\n");

      // reset this every 16th pass
      if ((uptime_cnt & 0xFU) == 0U) {
        pending_can_live = 0;
      }
      #ifdef DEBUG
        print("** blink ");
        print("rx:"); puth4(can_rx_q.r_ptr); print("-"); puth4(can_rx_q.w_ptr); print("  ");
        print("tx1:"); puth4(can_tx1_q.r_ptr); print("-"); puth4(can_tx1_q.w_ptr); print("  ");
        print("tx2:"); puth4(can_tx2_q.r_ptr); print("-"); puth4(can_tx2_q.w_ptr); print("  ");
        print("tx3:"); puth4(can_tx3_q.r_ptr); print("-"); puth4(can_tx3_q.w_ptr); print("\n");
      #endif

      // set green LED to be controls allowed
      current_board->set_led(LED_GREEN, controls_allowed | green_led_enabled);

      // turn off the blue LED, turned on by CAN
      // unless we are in power saving mode
      current_board->set_led(LED_BLUE, (uptime_cnt & 1U) && (power_save_status == POWER_SAVE_STATUS_ENABLED));

      // tick drivers at 1Hz
      logging_tick();

      const bool recent_heartbeat = heartbeat_counter == 0U;
      const bool harness_inserted = (harness.status != previous_harness_status) && (harness.status != HARNESS_STATUS_NC);
      const bool just_bootkicked = current_board->board_tick(check_started(), usb_enumerated, recent_heartbeat, harness_inserted);
      previous_harness_status = harness.status;

      // log device boot time
      const bool som_running = current_board->read_som_gpio();
      if (just_bootkicked && !som_running) {
        log("bootkick");
        waiting_to_boot = true;
      }
      if (waiting_to_boot) {
        if (som_running) {
          log("device booted");
          waiting_to_boot = false;
        } else if (waiting_to_boot_count == 45U) {
          log("not booted after 45s");
        } else {

        }
        waiting_to_boot_count += 1U;
      }

      // increase heartbeat counter and cap it at the uint32 limit
      if (heartbeat_counter < __UINT32_MAX__) {
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
        controls_allowed_countdown = 30U;
      } else if (controls_allowed_countdown > 0U) {
        controls_allowed_countdown -= 1U;
      } else {

      }

      // exit controls allowed if unused by openpilot for a few seconds
      if (controls_allowed && !heartbeat_engaged) {
        heartbeat_engaged_mismatches += 1U;
        if (heartbeat_engaged_mismatches >= 3U) {
          controls_allowed = false;
        }
      } else {
        heartbeat_engaged_mismatches = 0U;
      }

      if (!heartbeat_disabled) {
        // if the heartbeat has been gone for a while, go to SILENT safety mode and enter power save
        if (heartbeat_counter >= (check_started() ? HEARTBEAT_IGNITION_CNT_ON : HEARTBEAT_IGNITION_CNT_OFF)) {
          print("device hasn't sent a heartbeat for 0x");
          puth(heartbeat_counter);
          print(" seconds. Safety is set to SILENT mode.\n");

          if (controls_allowed_countdown > 0U) {
            siren_countdown = 5U;
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

          if (power_save_status != POWER_SAVE_STATUS_ENABLED) {
            set_power_save_state(POWER_SAVE_STATUS_ENABLED);
          }

          // Also disable IR when the heartbeat goes missing
          current_board->set_ir_power(0U);

          // Run fan when device is up, but not talking to us
          // * bootloader enables the SOM GPIO on boot
          // * fallback to USB enumerated where supported
          bool enabled = usb_enumerated || current_board->read_som_gpio();
          fan_set_power(enabled ? 50U : 0U);
        }
      }

      // check registers
      check_registers();

      // set ignition_can to false after 2s of no CAN seen
      if (ignition_can_cnt > 2U) {
        ignition_can = false;
      }

      // on to the next one
      uptime_cnt += 1U;
      safety_mode_cnt += 1U;
      ignition_can_cnt += 1U;

      // synchronous safety check
      safety_tick(current_rx_checks);
    }

    loop_counter++;
    loop_counter %= 8U;
  }
  TICK_TIMER->SR = 0;
}

void EXTI_IRQ_Handler(void) {
  if (check_exti_irq()) {
    exti_irq_clear();
    clock_init();

    set_power_save_state(POWER_SAVE_STATUS_DISABLED);
    deepsleep_allowed = false;
    heartbeat_counter = 0U;
    usb_soft_disconnect(false);

    NVIC_EnableIRQ(TICK_TIMER_IRQ);
  }
}

uint8_t rtc_counter = 0;
void RTC_WKUP_IRQ_Handler(void) {
  exti_irq_clear();
  clock_init();

  rtc_counter++;
  if ((rtc_counter % 2U) == 0U) {
    current_board->set_led(LED_BLUE, false);
  } else {
    current_board->set_led(LED_BLUE, true);
  }

  if (rtc_counter == __UINT8_MAX__) {
    rtc_counter = 1U;
  }
}


int main(void) {
  // Init interrupt table
  init_interrupts(true);

  // shouldn't have interrupts here, but just in case
  disable_interrupts();

  // init early devices
  clock_init();
  peripherals_init();
  detect_board_type();
  // red+green leds enabled until succesful USB/SPI init, as a debug indicator
  current_board->set_led(LED_RED, true);
  current_board->set_led(LED_GREEN, true);
  adc_init();
  logging_init();

  // print hello
  print("\n\n\n************************ MAIN START ************************\n");

  // check for non-supported board types
  if(hw_type == HW_TYPE_UNKNOWN){
    print("Unsupported board type\n");
    while (1) { /* hang */ }
  }

  print("Config:\n");
  print("  Board type: "); print(current_board->board_type); print("\n");

  // init board
  current_board->init();

  // panda has an FPU, let's use it!
  enable_fpu();

  log("main start");

  if (current_board->has_lin) {
    // enable LIN
    uart_init(&uart_ring_lin1, 10400);
    UART5->CR2 |= USART_CR2_LINEN;
    uart_init(&uart_ring_lin2, 10400);
    USART3->CR2 |= USART_CR2_LINEN;
  }

  if (current_board->fan_max_rpm > 0U) {
    fan_init();
  }

  microsecond_timer_init();

  // init to SILENT and can silent
  set_safety_mode(SAFETY_SILENT, 0U);

  // enable CAN TXs
  current_board->enable_can_transceivers(true);

  // init watchdog for heartbeat loop, fed at 8Hz
  simple_watchdog_init(FAULT_HEARTBEAT_LOOP_WATCHDOG, (3U * 1000000U / 8U));

  // 8Hz timer
  REGISTER_INTERRUPT(TICK_TIMER_IRQ, tick_handler, 10U, FAULT_INTERRUPT_RATE_TICK)
  tick_timer_init();

#ifdef DEBUG
  print("DEBUG ENABLED\n");
#endif
  // enable USB (right before interrupts or enum can fail!)
  usb_init();

#ifdef ENABLE_SPI
  if (current_board->has_spi) {
    spi_init();
  }
#endif

  current_board->set_led(LED_RED, false);
  current_board->set_led(LED_GREEN, false);

  print("**** INTERRUPTS ON ****\n");
  enable_interrupts();

  // LED should keep on blinking all the time
  uint64_t cnt = 0;

  for (cnt=0;;cnt++) {
    if (power_save_status == POWER_SAVE_STATUS_DISABLED) {
      #ifdef DEBUG_FAULTS
      if (fault_status == FAULT_STATUS_NONE) {
      #endif
        // useful for debugging, fade breaks = panda is overloaded
        for (uint32_t fade = 0U; fade < MAX_LED_FADE; fade += 1U) {
          current_board->set_led(LED_RED, true);
          delay(fade >> 4);
          current_board->set_led(LED_RED, false);
          delay((MAX_LED_FADE - fade) >> 4);
        }

        for (uint32_t fade = MAX_LED_FADE; fade > 0U; fade -= 1U) {
          current_board->set_led(LED_RED, true);
          delay(fade >> 4);
          current_board->set_led(LED_RED, false);
          delay((MAX_LED_FADE - fade) >> 4);
        }

      #ifdef DEBUG_FAULTS
      } else {
          current_board->set_led(LED_RED, 1);
          delay(512000U);
          current_board->set_led(LED_RED, 0);
          delay(512000U);
        }
      #endif
    } else {
      if (deepsleep_allowed && !usb_enumerated && !check_started() && ignition_seen && (heartbeat_counter > 20U)) {
        usb_soft_disconnect(true);
        fan_set_power(0U);
        NVIC_DisableIRQ(TICK_TIMER_IRQ);
        delay(512000U);

        // Init IRQs for CAN transceiver and ignition line
        exti_irq_init();

        // Init RTC Wakeup event on EXTI22
        REGISTER_INTERRUPT(RTC_WKUP_IRQn, RTC_WKUP_IRQ_Handler, 10U, FAULT_INTERRUPT_RATE_EXTI)
        rtc_wakeup_init();

        // STOP mode
        SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
      }
      __WFI();
      SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
    }
  }

  return 0;
}
