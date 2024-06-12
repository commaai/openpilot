// ********************* Includes *********************
#include "board/config.h"

#include "board/safety.h"

#include "board/drivers/pwm.h"
#include "board/drivers/usb.h"

#include "board/early_init.h"
#include "board/provision.h"

#include "board/health.h"
#include "jungle_health.h"

#include "board/drivers/can_common.h"

#ifdef STM32H7
  #include "board/drivers/fdcan.h"
#else
  #include "board/drivers/bxcan.h"
#endif

#include "board/obj/gitversion.h"

#include "board/can_comms.h"
#include "main_comms.h"


// ********************* Serial debugging *********************

void debug_ring_callback(uart_ring *ring) {
  char rcv;
  while (get_char(ring, &rcv)) {
    (void)injectc(ring, rcv);
  }
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

// called at 8Hz
uint32_t loop_counter = 0U;
uint16_t button_press_cnt = 0U;
void tick_handler(void) {
  if (TICK_TIMER->SR != 0) {
    if (generated_can_traffic) {
      for (int i = 0; i < 3; i++) {
        if (can_health[i].transmit_error_cnt >= 128) {
          (void)llcan_init(CANIF_FROM_CAN_NUM(i));
        }
      }
    }

    // tick drivers at 8Hz
    usb_tick();

    // decimated to 1Hz
    if ((loop_counter % 8) == 0U) {
      #ifdef DEBUG
        print("** blink ");
        print("rx:"); puth4(can_rx_q.r_ptr); print("-"); puth4(can_rx_q.w_ptr); print("  ");
        print("tx1:"); puth4(can_tx1_q.r_ptr); print("-"); puth4(can_tx1_q.w_ptr); print("  ");
        print("tx2:"); puth4(can_tx2_q.r_ptr); print("-"); puth4(can_tx2_q.w_ptr); print("  ");
        print("tx3:"); puth4(can_tx3_q.r_ptr); print("-"); puth4(can_tx3_q.w_ptr); print("\n");
      #endif

      current_board->board_tick();

      // check registers
      check_registers();

      // turn off the blue LED, turned on by CAN
      current_board->set_led(LED_BLUE, false);

      // Blink and OBD CAN
#ifdef FINAL_PROVISIONING
      current_board->set_can_mode(can_mode == CAN_MODE_NORMAL ? CAN_MODE_OBD_CAN2 : CAN_MODE_NORMAL);
#endif

      // on to the next one
      uptime_cnt += 1U;
    }

    current_board->set_led(LED_GREEN, green_led_enabled);

    // Check on button
    bool current_button_status = current_board->get_button();

    if (current_button_status && button_press_cnt == 10) {
      current_board->set_panda_power(!panda_power);
    }

#ifdef FINAL_PROVISIONING
    // Ignition blinking
    uint8_t ignition_bitmask = 0U;
    for (uint8_t i = 0U; i < 6U; i++) {
      ignition_bitmask |= ((loop_counter % 12U) < ((uint32_t) i + 2U)) << i;
    }
    current_board->set_individual_ignition(ignition_bitmask);

    // SBU voltage reporting
    if (current_board->has_sbu_sense) {
      for (uint8_t i = 0U; i < 6U; i++) {
        CANPacket_t pkt = { 0 };
        pkt.data_len_code = 8U;
        pkt.addr = 0x100U + i;
        *(uint16_t *) &pkt.data[0] = current_board->get_sbu_mV(i + 1U, SBU1);
        *(uint16_t *) &pkt.data[2] = current_board->get_sbu_mV(i + 1U, SBU2);
        pkt.data[4] = (ignition_bitmask >> i) & 1U;
        can_set_checksum(&pkt);
        can_send(&pkt, 0U, false);
      }
    }
#else
    // toggle ignition on button press
    static bool prev_button_status = false;
    if (!current_button_status && prev_button_status && button_press_cnt < 10){
      current_board->set_ignition(!ignition);
    }
    prev_button_status = current_button_status;
#endif

    button_press_cnt = current_button_status ? button_press_cnt + 1 : 0;

    loop_counter++;
  }
  TICK_TIMER->SR = 0;
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
  // red+green leds enabled until succesful USB init, as a debug indicator
  current_board->set_led(LED_RED, true);
  current_board->set_led(LED_GREEN, true);

  // print hello
  print("\n\n\n************************ MAIN START ************************\n");

  // check for non-supported board types
  if (hw_type == HW_TYPE_UNKNOWN) {
    print("Unsupported board type\n");
    while (1) { /* hang */ }
  }

  print("Config:\n");
  print("  Board type: 0x"); puth(hw_type); print("\n");

  // init board
  current_board->init();

  // we have an FPU, let's use it!
  enable_fpu();

  microsecond_timer_init();

  // 8Hz timer
  REGISTER_INTERRUPT(TICK_TIMER_IRQ, tick_handler, 10U, FAULT_INTERRUPT_RATE_TICK)
  tick_timer_init();

#ifdef DEBUG
  print("DEBUG ENABLED\n");
#endif
  // enable USB (right before interrupts or enum can fail!)
  usb_init();

  current_board->set_led(LED_RED, false);
  current_board->set_led(LED_GREEN, false);

  print("**** INTERRUPTS ON ****\n");
  enable_interrupts();

  can_silent = ALL_CAN_LIVE;
  set_safety_hooks(SAFETY_ALLOUTPUT, 0U);

  can_init_all();
  current_board->set_harness_orientation(HARNESS_ORIENTATION_1);

#ifdef FINAL_PROVISIONING
  print("---- FINAL PROVISIONING BUILD ---- \n");
  can_set_forwarding(0, 2);
  can_set_forwarding(1, 2);
#endif

  // LED should keep on blinking all the time
  uint32_t cnt = 0;
  for (cnt=0;;cnt++) {
    if (generated_can_traffic) {
      // fill up all the queues
      can_ring *qs[] = {&can_tx1_q, &can_tx2_q, &can_tx3_q};
      for (int j = 0; j < 3; j++) {
        for (uint16_t n = 0U; n < can_slots_empty(qs[j]); n++) {
          uint16_t i = cnt % 100U;
          CANPacket_t to_send;
          to_send.returned = 0U;
          to_send.rejected = 0U;
          to_send.extended = 0U;
          to_send.addr = 0x200U + i;
          to_send.bus = i % 3U;
          to_send.data_len_code = i % 8U;
          (void)memcpy(to_send.data, "\xff\xff\xff\xff\xff\xff\xff\xff", dlc_to_len[to_send.data_len_code]);
          can_set_checksum(&to_send);

          can_send(&to_send, to_send.bus, true);
        }
      }

      delay(1000);
      continue;
    }

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
  }

  return 0;
}
