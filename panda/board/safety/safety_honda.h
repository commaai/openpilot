// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      accel rising edge
//      brake rising edge
//      brake > 0mph

const int HONDA_GAS_INTERCEPTOR_THRESHOLD = 328;  // ratio between offset and gain from dbc file
int honda_brake_prev = 0;
int honda_gas_prev = 0;
int honda_ego_speed = 0;
bool honda_bosch_hardware = false;
bool honda_alt_brake_msg = false;

static void honda_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {

  int addr = GET_ADDR(to_push);
  int len = GET_LEN(to_push);

  // sample speed
  if (addr == 0x158) {
    // first 2 bytes
    honda_ego_speed = to_push->RDLR & 0xFFFF;
  }

  // state machine to enter and exit controls
  // 0x1A6 for the ILX, 0x296 for the Civic Touring
  if ((addr == 0x1A6) || (addr == 0x296)) {
    int button = (to_push->RDLR & 0xE0) >> 5;
    switch (button) {
      case 2:  // cancel
        controls_allowed = 0;
        break;
      case 3:  // set
      case 4:  // resume
        controls_allowed = 1;
        break;
      default:
        break; // any other button is irrelevant
    }
  }

  // user brake signal on 0x17C reports applied brake from computer brake on accord
  // and crv, which prevents the usual brake safety from working correctly. these
  // cars have a signal on 0x1BE which only detects user's brake being applied so
  // in these cases, this is used instead.
  // most hondas: 0x17C bit 53
  // accord, crv: 0x1BE bit 4
  #define IS_USER_BRAKE_MSG(addr) (!honda_alt_brake_msg ? ((addr) == 0x17C) : ((addr) == 0x1BE))
  #define USER_BRAKE_VALUE(to_push)  (!honda_alt_brake_msg ? ((to_push)->RDHR & 0x200000)  : ((to_push)->RDLR & 0x10))
  // exit controls on rising edge of brake press or on brake press when
  // speed > 0
  bool is_user_brake_msg = IS_USER_BRAKE_MSG(addr);  // needed to enforce type
  if (is_user_brake_msg) {
    int brake = USER_BRAKE_VALUE(to_push);
    if (brake && (!(honda_brake_prev) || honda_ego_speed)) {
      controls_allowed = 0;
    }
    honda_brake_prev = brake;
  }

  // exit controls on rising edge of gas press if interceptor (0x201 w/ len = 6)
  // length check because bosch hardware also uses this id (0x201 w/ len = 8)
  if ((addr == 0x201) && (len == 6)) {
    gas_interceptor_detected = 1;
    int gas_interceptor = ((to_push->RDLR & 0xFF) << 8) | ((to_push->RDLR & 0xFF00) >> 8);
    if ((gas_interceptor > HONDA_GAS_INTERCEPTOR_THRESHOLD) &&
        (gas_interceptor_prev <= HONDA_GAS_INTERCEPTOR_THRESHOLD) &&
        long_controls_allowed) {
      controls_allowed = 0;
    }
    gas_interceptor_prev = gas_interceptor;
  }

  // exit controls on rising edge of gas press if no interceptor
  if (!gas_interceptor_detected) {
    if (addr == 0x17C) {
      int gas = to_push->RDLR & 0xFF;
      if (gas && !(honda_gas_prev) && long_controls_allowed) {
        controls_allowed = 0;
      }
      honda_gas_prev = gas;
    }
  }
}

// all commands: gas, brake and steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int honda_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  // disallow actuator commands if gas or brake (with vehicle moving) are pressed
  // and the the latching controls_allowed flag is True
  int pedal_pressed = honda_gas_prev || (gas_interceptor_prev > HONDA_GAS_INTERCEPTOR_THRESHOLD) ||
                      (honda_brake_prev && honda_ego_speed);
  bool current_controls_allowed = controls_allowed && !(pedal_pressed);

  // BRAKE: safety check
  if (addr == 0x1FA) {
    if (!current_controls_allowed || !long_controls_allowed) {
      if ((to_send->RDLR & 0xFFFF0000) != to_send->RDLR) {
        tx = 0;
      }
    }
    if ((to_send->RDLR & 0xFFFFFF3F) != to_send->RDLR) {
      tx = 0;
    }
  }

  // STEER: safety check
  if ((addr == 0xE4) || (addr == 0x194)) {
    if (!current_controls_allowed) {
      if ((to_send->RDLR & 0xFFFF0000) != to_send->RDLR) {
        tx = 0;
      }
    }
  }

  // GAS: safety check
  if (addr == 0x200) {
    if (!current_controls_allowed || !long_controls_allowed) {
      if ((to_send->RDLR & 0xFFFF0000) != to_send->RDLR) {
        tx = 0;
      }
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button in Bosch HW
  // ensuring that only the cancel button press is sent (VAL 2) when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  if ((addr == 0x296) && honda_bosch_hardware &&
      !current_controls_allowed && (bus == 0)) {
    if (((to_send->RDLR >> 5) & 0x7) != 2) {
      tx = 0;
    }
  }

  // 1 allows the message through
  return tx;
}

static void honda_init(int16_t param) {
  UNUSED(param);
  controls_allowed = 0;
  honda_bosch_hardware = false;
  honda_alt_brake_msg = false;
}

static void honda_bosch_init(int16_t param) {
  controls_allowed = 0;
  honda_bosch_hardware = true;
  // Checking for alternate brake override from safety parameter
  honda_alt_brake_msg = (param == 1) ? true : false;
}

static int honda_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  // fwd from car to camera. also fwd certain msgs from camera to car
  // 0xE4 is steering on all cars except CRV and RDX, 0x194 for CRV and RDX,
  // 0x1FA is brake control, 0x30C is acc hud, 0x33D is lkas hud,
  // 0x39f is radar hud
  int bus_fwd = -1;

  if (bus_num == 0) {
    bus_fwd = 2;
  }
  if (bus_num == 2) {
    // block stock lkas messages and stock acc messages (if OP is doing ACC)
    int addr = GET_ADDR(to_fwd);
    int is_lkas_msg = (addr == 0xE4) || (addr == 0x194) || (addr == 0x33D);
    int is_acc_msg = (addr == 0x1FA) || (addr == 0x30C) || (addr == 0x39F);
    int block_fwd = is_lkas_msg || (is_acc_msg && long_controls_allowed);
    if (!block_fwd) {
      bus_fwd = 0;
    }
  }
  return bus_fwd;
}

static int honda_bosch_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  int bus_fwd = -1;

  if (bus_num == 2) {
    bus_fwd = 1;
  }
  if (bus_num == 1)  {
    int addr = GET_ADDR(to_fwd);
    int is_lkas_msg = (addr == 0xE4) || (addr == 0x33D);
    if (!is_lkas_msg) {
      bus_fwd = 2;
    }
  }
  return bus_fwd;
}

const safety_hooks honda_hooks = {
  .init = honda_init,
  .rx = honda_rx_hook,
  .tx = honda_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = honda_fwd_hook,
};

const safety_hooks honda_bosch_hooks = {
  .init = honda_bosch_init,
  .rx = honda_rx_hook,
  .tx = honda_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = honda_bosch_fwd_hook,
};
