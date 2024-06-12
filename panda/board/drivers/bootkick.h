bool bootkick_ign_prev = false;
BootState boot_state = BOOT_BOOTKICK;
uint8_t bootkick_harness_status_prev = HARNESS_STATUS_NC;

uint8_t boot_reset_countdown = 0;
uint8_t waiting_to_boot_countdown = 0;
bool bootkick_reset_triggered = false;
uint16_t bootkick_last_serial_ptr = 0;

void bootkick_tick(bool ignition, bool recent_heartbeat) {
  BootState boot_state_prev = boot_state;
  const bool harness_inserted = (harness.status != bootkick_harness_status_prev) && (harness.status != HARNESS_STATUS_NC);

  if ((ignition && !bootkick_ign_prev) || harness_inserted) {
    // bootkick on rising edge of ignition or harness insertion
    boot_state = BOOT_BOOTKICK;
  } else if (recent_heartbeat) {
    // disable bootkick once openpilot is up
    boot_state = BOOT_STANDBY;
  } else {

  }

  /*
    Ensure SOM boots in case it goes into QDL mode. Reset behavior:
    * shouldn't trigger on the first boot after power-on
    * only try reset once per bootkick, i.e. don't keep trying until booted
    * only try once per panda boot, since openpilot will reset panda on startup
    * once BOOT_RESET is triggered, it stays until countdown is finished
  */
  if (!bootkick_reset_triggered && (boot_state == BOOT_BOOTKICK) && (boot_state_prev == BOOT_STANDBY)) {
    waiting_to_boot_countdown = 20U;
  }
  if (waiting_to_boot_countdown > 0U) {
    bool serial_activity = uart_ring_som_debug.w_ptr_tx != bootkick_last_serial_ptr;
    if (serial_activity || current_board->read_som_gpio() || (boot_state != BOOT_BOOTKICK)) {
      waiting_to_boot_countdown = 0U;
    } else {
      // try a reset
      if (waiting_to_boot_countdown == 1U) {
        boot_reset_countdown = 5U;
      }
    }
  }

  // handle reset state
  if (boot_reset_countdown > 0U) {
    boot_state = BOOT_RESET;
    bootkick_reset_triggered = true;
  } else {
    if (boot_state == BOOT_RESET) {
      boot_state = BOOT_BOOTKICK;
    }
  }

  // update state
  bootkick_ign_prev = ignition;
  bootkick_harness_status_prev = harness.status;
  bootkick_last_serial_ptr = uart_ring_som_debug.w_ptr_tx;
  if (waiting_to_boot_countdown > 0U) {
    waiting_to_boot_countdown--;
  }
  if (boot_reset_countdown > 0U) {
    boot_reset_countdown--;
  }
  current_board->set_bootkick(boot_state);
}
