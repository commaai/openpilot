// WARNING: To stay in compliance with the SIL2 rules laid out in STM UM1840, we should never implement any of the available hardware low power modes.
// See rule: CoU_3

#define POWER_SAVE_STATUS_DISABLED 0
#define POWER_SAVE_STATUS_ENABLED 1

int power_save_status = POWER_SAVE_STATUS_DISABLED;

void set_power_save_state(int state) {

  bool is_valid_state = (state == POWER_SAVE_STATUS_ENABLED) || (state == POWER_SAVE_STATUS_DISABLED);
  if (is_valid_state && (state != power_save_status)) {
    bool enable = false;
    if (state == POWER_SAVE_STATUS_ENABLED) {
      print("enable power savings\n");

      // Disable CAN interrupts
      if (harness.status == HARNESS_STATUS_FLIPPED) {
        llcan_irq_disable(cans[0]);
      } else {
        llcan_irq_disable(cans[2]);
      }
      llcan_irq_disable(cans[1]);
    } else {
      print("disable power savings\n");

      if (harness.status == HARNESS_STATUS_FLIPPED) {
        llcan_irq_enable(cans[0]);
      } else {
        llcan_irq_enable(cans[2]);
      }
      llcan_irq_enable(cans[1]);

      enable = true;
    }

    current_board->enable_can_transceivers(enable);

    if(current_board->has_hw_gmlan){
      // turn on GMLAN
      set_gpio_output(GPIOB, 14, enable);
      set_gpio_output(GPIOB, 15, enable);
    }

    // Switch off IR when in power saving
    if(!enable){
      current_board->set_ir_power(0U);
    }

    power_save_status = state;
  }
}
