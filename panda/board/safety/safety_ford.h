// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      accel rising edge
//      brake rising edge
//      brake > 0mph

int ford_brake_prev = 0;
int ford_gas_prev = 0;
int ford_is_moving = 0;

static void ford_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {

  if ((to_push->RIR>>21) == 0x217) {
    // wheel speeds are 14 bits every 16
    ford_is_moving = 0xFCFF & (to_push->RDLR | (to_push->RDLR >> 16) |
                               to_push->RDHR | (to_push->RDHR >> 16));
  }

  // state machine to enter and exit controls
  if ((to_push->RIR>>21) == 0x83) {
    int cancel = ((to_push->RDLR >> 8) & 0x1);
    int set_or_resume = (to_push->RDLR >> 28) & 0x3;
    if (cancel) {
      controls_allowed = 0;
    } else if (set_or_resume) {
      controls_allowed = 1;
    }
  }

  // exit controls on rising edge of brake press or on brake press when
  // speed > 0
  if ((to_push->RIR>>21) == 0x165) {
    int brake = to_push->RDLR & 0x20;
    if (brake && (!(ford_brake_prev) || ford_is_moving)) {
      controls_allowed = 0;
    }
    ford_brake_prev = brake;
  }

  // exit controls on rising edge of gas press
  if ((to_push->RIR>>21) == 0x204) {
    int gas = to_push->RDLR & 0xFF03;
    if (gas && !(ford_gas_prev)) {
      controls_allowed = 0;
    }
    ford_gas_prev = gas;
  }
}

// all commands: just steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int ford_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  // disallow actuator commands if gas or brake (with vehicle moving) are pressed
  // and the the latching controls_allowed flag is True
  int pedal_pressed = ford_gas_prev || (ford_brake_prev && ford_is_moving);
  int current_controls_allowed = controls_allowed && !(pedal_pressed);

  // STEER: safety check
  if ((to_send->RIR>>21) == 0x3CA) {
    if (current_controls_allowed) {
      // all messages are fine here
    } else {
      // bits 7-4 need to be 0xF to disallow lkas commands
      if (((to_send->RDLR >> 4) & 0xF) != 0xF) return 0;
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button
  // ensuring that set and resume aren't sent
  if ((to_send->RIR>>21) == 0x83) {
    if ((to_send->RDLR >> 28) & 0x3) return 0;
  }

  // 1 allows the message through
  return true;
}

const safety_hooks ford_hooks = {
  .init = nooutput_init,
  .rx = ford_rx_hook,
  .tx = ford_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = nooutput_fwd_hook,
};
