// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      accel rising edge
//      brake rising edge
//      brake > 0mph
const CanMsg HONDA_N_TX_MSGS[] = {{0xE4, 0, 5}, {0x194, 0, 4}, {0x1FA, 0, 8}, {0x200, 0, 6}, {0x30C, 0, 8}, {0x33D, 0, 5}};
const CanMsg HONDA_BOSCH_TX_MSGS[] = {{0xE4, 0, 5}, {0xE5, 0, 8}, {0x296, 1, 4}, {0x33D, 0, 5}, {0x33DA, 0, 5}, {0x33DB, 0, 8}};  // Bosch
const CanMsg HONDA_RADARLESS_TX_MSGS[] = {{0xE4, 0, 5}, {0x296, 2, 4}, {0x33D, 0, 8}};  // Bosch radarless
const CanMsg HONDA_BOSCH_LONG_TX_MSGS[] = {{0xE4, 1, 5}, {0x1DF, 1, 8}, {0x1EF, 1, 8}, {0x1FA, 1, 8}, {0x30C, 1, 8}, {0x33D, 1, 5}, {0x33DA, 1, 5}, {0x33DB, 1, 8}, {0x39F, 1, 8}, {0x18DAB0F1, 1, 8}};  // Bosch w/ gas and brakes

// panda interceptor threshold needs to be equivalent to openpilot threshold to avoid controls mismatches
// If thresholds are mismatched then it is possible for panda to see the gas fall and rise while openpilot is in the pre-enabled state
// Threshold calculated from DBC gains: round(((83.3 / 0.253984064) + (83.3 / 0.126992032)) / 2) = 492
const int HONDA_GAS_INTERCEPTOR_THRESHOLD = 492;
#define HONDA_GET_INTERCEPTOR(msg) (((GET_BYTE((msg), 0) << 8) + GET_BYTE((msg), 1) + (GET_BYTE((msg), 2) << 8) + GET_BYTE((msg), 3)) / 2U)  // avg between 2 tracks

const LongitudinalLimits HONDA_BOSCH_LONG_LIMITS = {
  .max_accel = 200,   // accel is used for brakes
  .min_accel = -350,

  .max_gas = 2000,
  .min_gas = -30000,
  .inactive_gas = -30000,
};

const LongitudinalLimits HONDA_NIDEC_LONG_LIMITS = {
  .max_gas = 198,  // 0xc6
  .max_brake = 255,

  .inactive_speed = 0,
};

// Nidec and bosch radarless has the powertrain bus on bus 0
AddrCheckStruct honda_common_addr_checks[] = {
  {.msg = {{0x1A6, 0, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 40000U},                   // SCM_BUTTONS
           {0x296, 0, 4, .check_checksum = true, .max_counter = 3U, .expected_timestep = 40000U}, { 0 }}},
  {.msg = {{0x158, 0, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 10000U}, { 0 }, { 0 }}},   // ENGINE_DATA
  {.msg = {{0x17C, 0, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 10000U}, { 0 }, { 0 }}},   // POWERTRAIN_DATA
  {.msg = {{0x326, 0, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 100000U}, { 0 }, { 0 }}},  // SCM_FEEDBACK
};
#define HONDA_COMMON_ADDR_CHECKS_LEN (sizeof(honda_common_addr_checks) / sizeof(honda_common_addr_checks[0]))

// For Nidecs with main on signal on an alternate msg
AddrCheckStruct honda_nidec_alt_addr_checks[] = {
  {.msg = {{0x1A6, 0, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 40000U},
           {0x296, 0, 4, .check_checksum = true, .max_counter = 3U, .expected_timestep = 40000U}, { 0 }}},
  {.msg = {{0x158, 0, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0x17C, 0, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
};
#define HONDA_NIDEC_ALT_ADDR_CHECKS_LEN (sizeof(honda_nidec_alt_addr_checks) / sizeof(honda_nidec_alt_addr_checks[0]))

// Bosch has pt on bus 1
AddrCheckStruct honda_bosch_addr_checks[] = {
  {.msg = {{0x296, 1, 4, .check_checksum = true, .max_counter = 3U, .expected_timestep = 40000U}, { 0 }, { 0 }}},
  {.msg = {{0x158, 1, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0x17C, 1, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 10000U},
           {0x1BE, 1, 3, .check_checksum = true, .max_counter = 3U, .expected_timestep = 20000U}, { 0 }}},
  {.msg = {{0x326, 1, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 100000U}, { 0 }, { 0 }}},
};
#define HONDA_BOSCH_ADDR_CHECKS_LEN (sizeof(honda_bosch_addr_checks) / sizeof(honda_bosch_addr_checks[0]))

const uint16_t HONDA_PARAM_ALT_BRAKE = 1;
const uint16_t HONDA_PARAM_BOSCH_LONG = 2;
const uint16_t HONDA_PARAM_NIDEC_ALT = 4;
const uint16_t HONDA_PARAM_RADARLESS = 8;

enum {
  HONDA_BTN_NONE = 0,
  HONDA_BTN_MAIN = 1,
  HONDA_BTN_CANCEL = 2,
  HONDA_BTN_SET = 3,
  HONDA_BTN_RESUME = 4,
};

int honda_brake = 0;
bool honda_brake_switch_prev = false;
bool honda_alt_brake_msg = false;
bool honda_fwd_brake = false;
bool honda_bosch_long = false;
bool honda_bosch_radarless = false;
enum {HONDA_NIDEC, HONDA_BOSCH} honda_hw = HONDA_NIDEC;
addr_checks honda_rx_checks = {honda_common_addr_checks, HONDA_COMMON_ADDR_CHECKS_LEN};


int honda_get_pt_bus(void) {
  return ((honda_hw == HONDA_BOSCH) && !honda_bosch_radarless) ? 1 : 0;
}

static uint32_t honda_get_checksum(CANPacket_t *to_push) {
  int checksum_byte = GET_LEN(to_push) - 1U;
  return (uint8_t)(GET_BYTE(to_push, checksum_byte)) & 0xFU;
}

static uint32_t honda_compute_checksum(CANPacket_t *to_push) {
  int len = GET_LEN(to_push);
  uint8_t checksum = 0U;
  unsigned int addr = GET_ADDR(to_push);
  while (addr > 0U) {
    checksum += (addr & 0xFU); addr >>= 4;
  }
  for (int j = 0; j < len; j++) {
    uint8_t byte = GET_BYTE(to_push, j);
    checksum += (byte & 0xFU) + (byte >> 4U);
    if (j == (len - 1)) {
      checksum -= (byte & 0xFU);  // remove checksum in message
    }
  }
  return (uint8_t)((8U - checksum) & 0xFU);
}

static uint8_t honda_get_counter(CANPacket_t *to_push) {
  int counter_byte = GET_LEN(to_push) - 1U;
  return ((uint8_t)(GET_BYTE(to_push, counter_byte)) >> 4U) & 0x3U;
}

static int honda_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &honda_rx_checks,
                                 honda_get_checksum, honda_compute_checksum, honda_get_counter);

  if (valid) {
    const bool pcm_cruise = ((honda_hw == HONDA_BOSCH) && !honda_bosch_long) || \
                            ((honda_hw == HONDA_NIDEC) && !gas_interceptor_detected);
    int pt_bus = honda_get_pt_bus();

    int addr = GET_ADDR(to_push);
    int len = GET_LEN(to_push);
    int bus = GET_BUS(to_push);

    // sample speed
    if (addr == 0x158) {
      // first 2 bytes
      vehicle_moving = GET_BYTE(to_push, 0) | GET_BYTE(to_push, 1);
    }

    // check ACC main state
    // 0x326 for all Bosch and some Nidec, 0x1A6 for some Nidec
    if ((addr == 0x326) || (addr == 0x1A6)) {
      acc_main_on = GET_BIT(to_push, ((addr == 0x326) ? 28U : 47U));
      if (!acc_main_on) {
        controls_allowed = 0;
      }
    }

    // enter controls when PCM enters cruise state
    if (pcm_cruise && (addr == 0x17C)) {
      const bool cruise_engaged = GET_BIT(to_push, 38U) != 0U;
      // engage on rising edge
      if (cruise_engaged && !cruise_engaged_prev) {
        controls_allowed = 1;
      }

      // Since some Nidec cars can brake down to 0 after the PCM disengages,
      // we don't disengage when the PCM does.
      if (!cruise_engaged && (honda_hw != HONDA_NIDEC)) {
        controls_allowed = 0;
      }
      cruise_engaged_prev = cruise_engaged;
    }

    // state machine to enter and exit controls for button enabling
    // 0x1A6 for the ILX, 0x296 for the Civic Touring
    if (((addr == 0x1A6) || (addr == 0x296)) && (bus == pt_bus)) {
      int button = (GET_BYTE(to_push, 0) & 0xE0U) >> 5;

      // exit controls once main or cancel are pressed
      if ((button == HONDA_BTN_MAIN) || (button == HONDA_BTN_CANCEL)) {
        controls_allowed = 0;
      }

      // enter controls on the falling edge of set or resume
      bool set = (button == HONDA_BTN_NONE) && (cruise_button_prev == HONDA_BTN_SET);
      bool res = (button == HONDA_BTN_NONE) && (cruise_button_prev == HONDA_BTN_RESUME);
      if (acc_main_on && !pcm_cruise && (set || res)) {
        controls_allowed = 1;
      }
      cruise_button_prev = button;
    }

    // user brake signal on 0x17C reports applied brake from computer brake on accord
    // and crv, which prevents the usual brake safety from working correctly. these
    // cars have a signal on 0x1BE which only detects user's brake being applied so
    // in these cases, this is used instead.
    // most hondas: 0x17C
    // accord, crv: 0x1BE
    if (honda_alt_brake_msg) {
      if (addr == 0x1BE) {
        brake_pressed = GET_BIT(to_push, 4U) != 0U;
      }
    } else {
      if (addr == 0x17C) {
        // also if brake switch is 1 for two CAN frames, as brake pressed is delayed
        const bool brake_switch = GET_BIT(to_push, 32U) != 0U;
        brake_pressed = (GET_BIT(to_push, 53U) != 0U) || (brake_switch && honda_brake_switch_prev);
        honda_brake_switch_prev = brake_switch;
      }
    }

    // length check because bosch hardware also uses this id (0x201 w/ len = 8)
    if ((addr == 0x201) && (len == 6)) {
      gas_interceptor_detected = 1;
      int gas_interceptor = HONDA_GET_INTERCEPTOR(to_push);
      gas_pressed = gas_interceptor > HONDA_GAS_INTERCEPTOR_THRESHOLD;
      gas_interceptor_prev = gas_interceptor;
    }

    if (!gas_interceptor_detected) {
      if (addr == 0x17C) {
        gas_pressed = GET_BYTE(to_push, 0) != 0U;
      }
    }

    // disable stock Honda AEB in alternative experience
    if (!(alternative_experience & ALT_EXP_DISABLE_STOCK_AEB)) {
      if ((bus == 2) && (addr == 0x1FA)) {
        bool honda_stock_aeb = GET_BYTE(to_push, 3) & 0x20U;
        int honda_stock_brake = (GET_BYTE(to_push, 0) << 2) + ((GET_BYTE(to_push, 1) >> 6) & 0x3U);

        // Forward AEB when stock braking is higher than openpilot braking
        // only stop forwarding when AEB event is over
        if (!honda_stock_aeb) {
          honda_fwd_brake = false;
        } else if (honda_stock_brake >= honda_brake) {
          honda_fwd_brake = true;
        } else {
          // Leave Honda forward brake as is
        }
      }
    }

    int bus_rdr_car = (honda_hw == HONDA_BOSCH) ? 0 : 2;  // radar bus, car side
    bool stock_ecu_detected = false;

    if (safety_mode_cnt > RELAY_TRNS_TIMEOUT) {
      // If steering controls messages are received on the destination bus, it's an indication
      // that the relay might be malfunctioning
      if ((addr == 0xE4) || (addr == 0x194)) {
        if (((honda_hw != HONDA_NIDEC) && (bus == bus_rdr_car)) || ((honda_hw == HONDA_NIDEC) && (bus == 0))) {
          stock_ecu_detected = true;
        }
      }
      // If Honda Bosch longitudinal mode is selected we need to ensure the radar is turned off
      // Verify this by ensuring ACC_CONTROL (0x1DF) is not received on the PT bus
      if (honda_bosch_long && !honda_bosch_radarless && (bus == pt_bus) && (addr == 0x1DF)) {
        stock_ecu_detected = true;
      }
    }

    generic_rx_checks(stock_ecu_detected);
  }
  return valid;
}

// all commands: gas, brake and steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int honda_tx_hook(CANPacket_t *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if ((honda_hw == HONDA_BOSCH) && honda_bosch_radarless) {
    tx = msg_allowed(to_send, HONDA_RADARLESS_TX_MSGS, sizeof(HONDA_RADARLESS_TX_MSGS)/sizeof(HONDA_RADARLESS_TX_MSGS[0]));
  } else if ((honda_hw == HONDA_BOSCH) && !honda_bosch_long) {
    tx = msg_allowed(to_send, HONDA_BOSCH_TX_MSGS, sizeof(HONDA_BOSCH_TX_MSGS)/sizeof(HONDA_BOSCH_TX_MSGS[0]));
  } else if ((honda_hw == HONDA_BOSCH) && honda_bosch_long) {
    tx = msg_allowed(to_send, HONDA_BOSCH_LONG_TX_MSGS, sizeof(HONDA_BOSCH_LONG_TX_MSGS)/sizeof(HONDA_BOSCH_LONG_TX_MSGS[0]));
  } else {
    tx = msg_allowed(to_send, HONDA_N_TX_MSGS, sizeof(HONDA_N_TX_MSGS)/sizeof(HONDA_N_TX_MSGS[0]));
  }

  int bus_pt = honda_get_pt_bus();
  int bus_buttons = (honda_bosch_radarless) ? 2 : bus_pt;  // the camera controls ACC on radarless Bosch cars

  // ACC_HUD: safety check (nidec w/o pedal)
  if ((addr == 0x30C) && (bus == bus_pt)) {
    int pcm_speed = (GET_BYTE(to_send, 0) << 8) | GET_BYTE(to_send, 1);
    int pcm_gas = GET_BYTE(to_send, 2);

    bool violation = false;
    violation |= longitudinal_speed_checks(pcm_speed, HONDA_NIDEC_LONG_LIMITS);
    violation |= longitudinal_gas_checks(pcm_gas, HONDA_NIDEC_LONG_LIMITS);
    if (violation) {
      tx = 0;
    }
  }

  // BRAKE: safety check (nidec)
  if ((addr == 0x1FA) && (bus == bus_pt)) {
    honda_brake = (GET_BYTE(to_send, 0) << 2) + ((GET_BYTE(to_send, 1) >> 6) & 0x3U);
    if (longitudinal_brake_checks(honda_brake, HONDA_NIDEC_LONG_LIMITS)) {
      tx = 0;
    }
    if (honda_fwd_brake) {
      tx = 0;
    }
  }

  // BRAKE/GAS: safety check (bosch)
  if ((addr == 0x1DF) && (bus == bus_pt)) {
    int accel = (GET_BYTE(to_send, 3) << 3) | ((GET_BYTE(to_send, 4) >> 5) & 0x7U);
    accel = to_signed(accel, 11);

    int gas = (GET_BYTE(to_send, 0) << 8) | GET_BYTE(to_send, 1);
    gas = to_signed(gas, 16);

    bool violation = false;
    violation |= longitudinal_accel_checks(accel, HONDA_BOSCH_LONG_LIMITS);
    violation |= longitudinal_gas_checks(gas, HONDA_BOSCH_LONG_LIMITS);
    if (violation) {
      tx = 0;
    }
  }

  // STEER: safety check
  if ((addr == 0xE4) || (addr == 0x194)) {
    if (!controls_allowed) {
      bool steer_applied = GET_BYTE(to_send, 0) | GET_BYTE(to_send, 1);
      if (steer_applied) {
        tx = 0;
      }
    }
  }

  // Bosch supplemental control check
  if (addr == 0xE5) {
    if ((GET_BYTES_04(to_send) != 0x10800004U) || ((GET_BYTES_48(to_send) & 0x00FFFFFFU) != 0x0U)) {
      tx = 0;
    }
  }

  // GAS: safety check (interceptor)
  if (addr == 0x200) {
    if (longitudinal_interceptor_checks(to_send)) {
      tx = 0;
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button in Bosch HW
  // ensuring that only the cancel button press is sent (VAL 2) when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  if ((addr == 0x296) && !controls_allowed && (bus == bus_buttons)) {
    if (((GET_BYTE(to_send, 0) >> 5) & 0x7U) != 2U) {
      tx = 0;
    }
  }

  // Only tester present ("\x02\x3E\x80\x00\x00\x00\x00\x00") allowed on diagnostics address
  if (addr == 0x18DAB0F1) {
    if ((GET_BYTES_04(to_send) != 0x00803E02U) || (GET_BYTES_48(to_send) != 0x0U)) {
      tx = 0;
    }
  }

  // 1 allows the message through
  return tx;
}

static const addr_checks* honda_nidec_init(uint16_t param) {
  gas_interceptor_detected = 0;
  honda_hw = HONDA_NIDEC;
  honda_alt_brake_msg = false;
  honda_bosch_long = false;
  honda_bosch_radarless = false;

  if (GET_FLAG(param, HONDA_PARAM_NIDEC_ALT)) {
    honda_rx_checks = (addr_checks){honda_nidec_alt_addr_checks, HONDA_NIDEC_ALT_ADDR_CHECKS_LEN};
  } else {
    honda_rx_checks = (addr_checks){honda_common_addr_checks, HONDA_COMMON_ADDR_CHECKS_LEN};
  }
  return &honda_rx_checks;
}

static const addr_checks* honda_bosch_init(uint16_t param) {
  honda_hw = HONDA_BOSCH;
  honda_bosch_radarless = GET_FLAG(param, HONDA_PARAM_RADARLESS);
  // Checking for alternate brake override from safety parameter
  honda_alt_brake_msg = GET_FLAG(param, HONDA_PARAM_ALT_BRAKE) && !honda_bosch_radarless;

  // radar disabled so allow gas/brakes
#ifdef ALLOW_DEBUG
  honda_bosch_long = GET_FLAG(param, HONDA_PARAM_BOSCH_LONG) && !honda_bosch_radarless;
#endif

  if (honda_bosch_radarless) {
    honda_rx_checks = (addr_checks){honda_common_addr_checks, HONDA_COMMON_ADDR_CHECKS_LEN};
  } else {
    honda_rx_checks = (addr_checks){honda_bosch_addr_checks, HONDA_BOSCH_ADDR_CHECKS_LEN};
  }
  return &honda_rx_checks;
}

static int honda_nidec_fwd_hook(int bus_num, CANPacket_t *to_fwd) {
  // fwd from car to camera. also fwd certain msgs from camera to car
  // 0xE4 is steering on all cars except CRV and RDX, 0x194 for CRV and RDX,
  // 0x1FA is brake control, 0x30C is acc hud, 0x33D is lkas hud
  int bus_fwd = -1;

  if (bus_num == 0) {
    bus_fwd = 2;
  }

  if (bus_num == 2) {
    // block stock lkas messages and stock acc messages (if OP is doing ACC)
    int addr = GET_ADDR(to_fwd);
    bool is_lkas_msg = (addr == 0xE4) || (addr == 0x194) || (addr == 0x33D);
    bool is_acc_hud_msg = addr == 0x30C;
    bool is_brake_msg = addr == 0x1FA;
    bool block_fwd = is_lkas_msg || is_acc_hud_msg || (is_brake_msg && !honda_fwd_brake);
    if (!block_fwd) {
      bus_fwd = 0;
    }
  }

  return bus_fwd;
}

static int honda_bosch_fwd_hook(int bus_num, CANPacket_t *to_fwd) {
  int bus_fwd = -1;

  if (bus_num == 0) {
    bus_fwd = 2;
  }
  if (bus_num == 2)  {
    int addr = GET_ADDR(to_fwd);
    int is_lkas_msg = (addr == 0xE4) || (addr == 0xE5) || (addr == 0x33D) || (addr == 0x33DA) || (addr == 0x33DB);
    if (!is_lkas_msg) {
      bus_fwd = 0;
    }
  }

  return bus_fwd;
}

const safety_hooks honda_nidec_hooks = {
  .init = honda_nidec_init,
  .rx = honda_rx_hook,
  .tx = honda_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = honda_nidec_fwd_hook,
};

const safety_hooks honda_bosch_hooks = {
  .init = honda_bosch_init,
  .rx = honda_rx_hook,
  .tx = honda_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = honda_bosch_fwd_hook,
};
