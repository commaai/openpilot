// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      accel rising edge
//      brake rising edge
//      brake > 0mph


static int ford_rx_hook(CANPacket_t *to_push) {

  int addr = GET_ADDR(to_push);
  int bus = GET_BUS(to_push);
  bool unsafe_allow_gas = unsafe_mode & UNSAFE_DISABLE_DISENGAGE_ON_GAS;

  if (addr == 0x217) {
    // wheel speeds are 14 bits every 16
    vehicle_moving = false;
    for (int i = 0; i < 8; i += 2) {
      vehicle_moving |= GET_BYTE(to_push, i) | (GET_BYTE(to_push, (int)(i + 1)) & 0xFCU);
    }
  }

  // state machine to enter and exit controls
  if (addr == 0x83) {
    bool cancel = GET_BYTE(to_push, 1) & 0x1;
    bool set_or_resume = GET_BYTE(to_push, 3) & 0x30;
    if (cancel) {
      controls_allowed = 0;
    }
    if (set_or_resume) {
      controls_allowed = 1;
    }
  }

  // exit controls on rising edge of brake press or on brake press when
  // speed > 0
  if (addr == 0x165) {
    brake_pressed = GET_BYTE(to_push, 0) & 0x20;
    if (brake_pressed && (!brake_pressed_prev || vehicle_moving)) {
      controls_allowed = 0;
    }
    brake_pressed_prev = brake_pressed;
  }

  // exit controls on rising edge of gas press
  if (addr == 0x204) {
    gas_pressed = ((GET_BYTE(to_push, 0) & 0x03) | GET_BYTE(to_push, 1)) != 0;
    if (!unsafe_allow_gas && gas_pressed && !gas_pressed_prev) {
      controls_allowed = 0;
    }
    gas_pressed_prev = gas_pressed;
  }

  if ((safety_mode_cnt > RELAY_TRNS_TIMEOUT) && (bus == 0) && (addr == 0x3CA)) {
    relay_malfunction_set();
  }
  return 1;
}

// all commands: just steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int ford_tx_hook(CANPacket_t *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);

  // disallow actuator commands if gas or brake (with vehicle moving) are pressed
  // and the the latching controls_allowed flag is True
  int pedal_pressed = brake_pressed_prev && vehicle_moving;
  bool unsafe_allow_gas = unsafe_mode & UNSAFE_DISABLE_DISENGAGE_ON_GAS;
  if (!unsafe_allow_gas) {
    pedal_pressed = pedal_pressed || gas_pressed_prev;
  }
  bool current_controls_allowed = controls_allowed && !(pedal_pressed);

  // STEER: safety check
  if (addr == 0x3CA) {
    if (!current_controls_allowed) {
      // bits 7-4 need to be 0xF to disallow lkas commands
      if ((GET_BYTE(to_send, 0) & 0xF0) != 0xF0) {
        tx = 0;
      }
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button
  // ensuring that set and resume aren't sent
  if (addr == 0x83) {
    if ((GET_BYTE(to_send, 3) & 0x30) != 0) {
      tx = 0;
    }
  }

  // 1 allows the message through
  return tx;
}

// TODO: keep camera on bus 2 and make a fwd_hook

const safety_hooks ford_hooks = {
  .init = nooutput_init,
  .rx = ford_rx_hook,
  .tx = ford_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = default_fwd_hook,
};
