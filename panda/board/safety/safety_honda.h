const CanMsg HONDA_N_TX_MSGS[] = {{0xE4, 0, 5}, {0x194, 0, 4}, {0x1FA, 0, 8}, {0x30C, 0, 8}, {0x33D, 0, 5}};
const CanMsg HONDA_N_INTERCEPTOR_TX_MSGS[] = {{0xE4, 0, 5}, {0x194, 0, 4}, {0x1FA, 0, 8}, {0x200, 0, 6}, {0x30C, 0, 8}, {0x33D, 0, 5}};
const CanMsg HONDA_BOSCH_TX_MSGS[] = {{0xE4, 0, 5}, {0xE5, 0, 8}, {0x296, 1, 4}, {0x33D, 0, 5}, {0x33DA, 0, 5}, {0x33DB, 0, 8}};  // Bosch
const CanMsg HONDA_BOSCH_LONG_TX_MSGS[] = {{0xE4, 1, 5}, {0x1DF, 1, 8}, {0x1EF, 1, 8}, {0x1FA, 1, 8}, {0x30C, 1, 8}, {0x33D, 1, 5}, {0x33DA, 1, 5}, {0x33DB, 1, 8}, {0x39F, 1, 8}, {0x18DAB0F1, 1, 8}};  // Bosch w/ gas and brakes
const CanMsg HONDA_RADARLESS_TX_MSGS[] = {{0xE4, 0, 5}, {0x296, 2, 4}, {0x33D, 0, 8}};  // Bosch radarless
const CanMsg HONDA_RADARLESS_LONG_TX_MSGS[] = {{0xE4, 0, 5}, {0x33D, 0, 8}, {0x1C8, 0, 8}, {0x30C, 0, 8}};  // Bosch radarless w/ gas and brakes

// panda interceptor threshold needs to be equivalent to openpilot threshold to avoid controls mismatches
// If thresholds are mismatched then it is possible for panda to see the gas fall and rise while openpilot is in the pre-enabled state
// Threshold calculated from DBC gains: round(((83.3 / 0.253984064) + (83.3 / 0.126992032)) / 2) = 492
const int HONDA_GAS_INTERCEPTOR_THRESHOLD = 492;
#define HONDA_GET_INTERCEPTOR(msg) (((GET_BYTE((msg), 0) << 8) + GET_BYTE((msg), 1) + (GET_BYTE((msg), 2) << 8) + GET_BYTE((msg), 3)) / 2U)  // avg between 2 tracks

const LongitudinalLimits HONDA_BOSCH_LONG_LIMITS = {
  .max_accel = 200,   // accel is used for brakes
  .min_accel = -350,

  .max_gas = 2000,
  .inactive_gas = -30000,
};

const LongitudinalLimits HONDA_NIDEC_LONG_LIMITS = {
  .max_gas = 198,  // 0xc6
  .max_brake = 255,

  .inactive_speed = 0,
};

// All common address checks except SCM_BUTTONS which isn't on one Nidec safety configuration
#define HONDA_COMMON_NO_SCM_FEEDBACK_RX_CHECKS(pt_bus)                                                                                           \
  {.msg = {{0x1A6, (pt_bus), 8, .check_checksum = true, .max_counter = 3U, .frequency = 25U},                  /* SCM_BUTTONS */      \
           {0x296, (pt_bus), 4, .check_checksum = true, .max_counter = 3U, .frequency = 25U}, { 0 }}},                                \
  {.msg = {{0x158, (pt_bus), 8, .check_checksum = true, .max_counter = 3U, .frequency = 100U}, { 0 }, { 0 }}},  /* ENGINE_DATA */      \
  {.msg = {{0x17C, (pt_bus), 8, .check_checksum = true, .max_counter = 3U, .frequency = 100U}, { 0 }, { 0 }}},  /* POWERTRAIN_DATA */  \

#define HONDA_COMMON_RX_CHECKS(pt_bus)                                                                                                         \
  HONDA_COMMON_NO_SCM_FEEDBACK_RX_CHECKS(pt_bus)                                                                                               \
  {.msg = {{0x326, (pt_bus), 8, .check_checksum = true, .max_counter = 3U, .frequency = 10U}, { 0 }, { 0 }}},  /* SCM_FEEDBACK */  \

// Alternate brake message is used on some Honda Bosch, and Honda Bosch radarless (where PT bus is 0)
#define HONDA_ALT_BRAKE_ADDR_CHECK(pt_bus)                                                                                                    \
  {.msg = {{0x1BE, (pt_bus), 3, .check_checksum = true, .max_counter = 3U, .frequency = 50U}, { 0 }, { 0 }}},  /* BRAKE_MODULE */  \


// Nidec and bosch radarless has the powertrain bus on bus 0
RxCheck honda_common_rx_checks[] = {
  HONDA_COMMON_RX_CHECKS(0)
};

RxCheck honda_common_interceptor_rx_checks[] = {
  HONDA_COMMON_RX_CHECKS(0)
  {.msg = {{0x201, 0, 6, .check_checksum = false, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
};

RxCheck honda_common_alt_brake_rx_checks[] = {
  HONDA_COMMON_RX_CHECKS(0)
  HONDA_ALT_BRAKE_ADDR_CHECK(0)
};

// For Nidecs with main on signal on an alternate msg (missing 0x326)
RxCheck honda_nidec_alt_rx_checks[] = {
  HONDA_COMMON_NO_SCM_FEEDBACK_RX_CHECKS(0)
};

RxCheck honda_nidec_alt_interceptor_rx_checks[] = {
  HONDA_COMMON_NO_SCM_FEEDBACK_RX_CHECKS(0)
  {.msg = {{0x201, 0, 6, .check_checksum = false, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
};

// Bosch has pt on bus 1, verified 0x1A6 does not exist
RxCheck honda_bosch_rx_checks[] = {
  HONDA_COMMON_RX_CHECKS(1)
};

RxCheck honda_bosch_alt_brake_rx_checks[] = {
  HONDA_COMMON_RX_CHECKS(1)
  HONDA_ALT_BRAKE_ADDR_CHECK(1)
};

const uint16_t HONDA_PARAM_ALT_BRAKE = 1;
const uint16_t HONDA_PARAM_BOSCH_LONG = 2;
const uint16_t HONDA_PARAM_NIDEC_ALT = 4;
const uint16_t HONDA_PARAM_RADARLESS = 8;
const uint16_t HONDA_PARAM_GAS_INTERCEPTOR = 16;

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


int honda_get_pt_bus(void) {
  return ((honda_hw == HONDA_BOSCH) && !honda_bosch_radarless) ? 1 : 0;
}

static uint32_t honda_get_checksum(const CANPacket_t *to_push) {
  int checksum_byte = GET_LEN(to_push) - 1U;
  return (uint8_t)(GET_BYTE(to_push, checksum_byte)) & 0xFU;
}

static uint32_t honda_compute_checksum(const CANPacket_t *to_push) {
  int len = GET_LEN(to_push);
  uint8_t checksum = 0U;
  unsigned int addr = GET_ADDR(to_push);
  while (addr > 0U) {
    checksum += (uint8_t)(addr & 0xFU); addr >>= 4;
  }
  for (int j = 0; j < len; j++) {
    uint8_t byte = GET_BYTE(to_push, j);
    checksum += (uint8_t)(byte & 0xFU) + (byte >> 4U);
    if (j == (len - 1)) {
      checksum -= (byte & 0xFU);  // remove checksum in message
    }
  }
  return (uint8_t)((8U - checksum) & 0xFU);
}

static uint8_t honda_get_counter(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t cnt = 0U;
  if (addr == 0x201) {
    // Signal: COUNTER_PEDAL
    cnt = GET_BYTE(to_push, 4) & 0x0FU;
  } else {
    int counter_byte = GET_LEN(to_push) - 1U;
    cnt = (GET_BYTE(to_push, counter_byte) >> 4U) & 0x3U;
  }
  return cnt;
}

static void honda_rx_hook(const CANPacket_t *to_push) {
  const bool pcm_cruise = ((honda_hw == HONDA_BOSCH) && !honda_bosch_long) || \
                          ((honda_hw == HONDA_NIDEC) && !enable_gas_interceptor);
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
      controls_allowed = false;
    }
  }

  // enter controls when PCM enters cruise state
  if (pcm_cruise && (addr == 0x17C)) {
    const bool cruise_engaged = GET_BIT(to_push, 38U);
    // engage on rising edge
    if (cruise_engaged && !cruise_engaged_prev) {
      controls_allowed = true;
    }

    // Since some Nidec cars can brake down to 0 after the PCM disengages,
    // we don't disengage when the PCM does.
    if (!cruise_engaged && (honda_hw != HONDA_NIDEC)) {
      controls_allowed = false;
    }
    cruise_engaged_prev = cruise_engaged;
  }

  // state machine to enter and exit controls for button enabling
  // 0x1A6 for the ILX, 0x296 for the Civic Touring
  if (((addr == 0x1A6) || (addr == 0x296)) && (bus == pt_bus)) {
    int button = (GET_BYTE(to_push, 0) & 0xE0U) >> 5;

    // enter controls on the falling edge of set or resume
    bool set = (button != HONDA_BTN_SET) && (cruise_button_prev == HONDA_BTN_SET);
    bool res = (button != HONDA_BTN_RESUME) && (cruise_button_prev == HONDA_BTN_RESUME);
    if (acc_main_on && !pcm_cruise && (set || res)) {
      controls_allowed = true;
    }

    // exit controls once main or cancel are pressed
    if ((button == HONDA_BTN_MAIN) || (button == HONDA_BTN_CANCEL)) {
      controls_allowed = false;
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
      brake_pressed = GET_BIT(to_push, 4U);
    }
  } else {
    if (addr == 0x17C) {
      // also if brake switch is 1 for two CAN frames, as brake pressed is delayed
      const bool brake_switch = GET_BIT(to_push, 32U);
      brake_pressed = (GET_BIT(to_push, 53U)) || (brake_switch && honda_brake_switch_prev);
      honda_brake_switch_prev = brake_switch;
    }
  }

  // length check because bosch hardware also uses this id (0x201 w/ len = 8)
  if ((addr == 0x201) && (len == 6) && enable_gas_interceptor) {
    int gas_interceptor = HONDA_GET_INTERCEPTOR(to_push);
    gas_pressed = gas_interceptor > HONDA_GAS_INTERCEPTOR_THRESHOLD;
    gas_interceptor_prev = gas_interceptor;
  }

  if (!enable_gas_interceptor) {
    if (addr == 0x17C) {
      gas_pressed = GET_BYTE(to_push, 0) != 0U;
    }
  }

  // disable stock Honda AEB in alternative experience
  if (!(alternative_experience & ALT_EXP_DISABLE_STOCK_AEB)) {
    if ((bus == 2) && (addr == 0x1FA)) {
      bool honda_stock_aeb = GET_BIT(to_push, 29U);
      int honda_stock_brake = (GET_BYTE(to_push, 0) << 2) | (GET_BYTE(to_push, 1) >> 6);

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

  generic_rx_checks(stock_ecu_detected);

}

static bool honda_tx_hook(const CANPacket_t *to_send) {
  bool tx = true;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

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
      tx = false;
    }
  }

  // BRAKE: safety check (nidec)
  if ((addr == 0x1FA) && (bus == bus_pt)) {
    honda_brake = (GET_BYTE(to_send, 0) << 2) + ((GET_BYTE(to_send, 1) >> 6) & 0x3U);
    if (longitudinal_brake_checks(honda_brake, HONDA_NIDEC_LONG_LIMITS)) {
      tx = false;
    }
    if (honda_fwd_brake) {
      tx = false;
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
      tx = false;
    }
  }

  // ACCEL: safety check (radarless)
  if ((addr == 0x1C8) && (bus == bus_pt)) {
    int accel = (GET_BYTE(to_send, 0) << 4) | (GET_BYTE(to_send, 1) >> 4);
    accel = to_signed(accel, 12);

    bool violation = false;
    violation |= longitudinal_accel_checks(accel, HONDA_BOSCH_LONG_LIMITS);
    if (violation) {
      tx = false;
    }
  }

  // STEER: safety check
  if ((addr == 0xE4) || (addr == 0x194)) {
    if (!controls_allowed) {
      bool steer_applied = GET_BYTE(to_send, 0) | GET_BYTE(to_send, 1);
      if (steer_applied) {
        tx = false;
      }
    }
  }

  // Bosch supplemental control check
  if (addr == 0xE5) {
    if ((GET_BYTES(to_send, 0, 4) != 0x10800004U) || ((GET_BYTES(to_send, 4, 4) & 0x00FFFFFFU) != 0x0U)) {
      tx = false;
    }
  }

  // GAS: safety check (interceptor)
  if (addr == 0x200) {
    if (longitudinal_interceptor_checks(to_send)) {
      tx = false;
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button in Bosch HW
  // ensuring that only the cancel button press is sent (VAL 2) when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  if ((addr == 0x296) && !controls_allowed && (bus == bus_buttons)) {
    if (((GET_BYTE(to_send, 0) >> 5) & 0x7U) != 2U) {
      tx = false;
    }
  }

  // Only tester present ("\x02\x3E\x80\x00\x00\x00\x00\x00") allowed on diagnostics address
  if (addr == 0x18DAB0F1) {
    if ((GET_BYTES(to_send, 0, 4) != 0x00803E02U) || (GET_BYTES(to_send, 4, 4) != 0x0U)) {
      tx = false;
    }
  }

  return tx;
}

static safety_config honda_nidec_init(uint16_t param) {
  honda_hw = HONDA_NIDEC;
  honda_brake = 0;
  honda_brake_switch_prev = false;
  honda_fwd_brake = false;
  honda_alt_brake_msg = false;
  honda_bosch_long = false;
  honda_bosch_radarless = false;
  enable_gas_interceptor = GET_FLAG(param, HONDA_PARAM_GAS_INTERCEPTOR);

  safety_config ret;

  bool enable_nidec_alt = GET_FLAG(param, HONDA_PARAM_NIDEC_ALT);
  if (enable_nidec_alt) {
    enable_gas_interceptor ? SET_RX_CHECKS(honda_nidec_alt_interceptor_rx_checks, ret) : \
                             SET_RX_CHECKS(honda_nidec_alt_rx_checks, ret);
  } else {
    enable_gas_interceptor ? SET_RX_CHECKS(honda_common_interceptor_rx_checks, ret) : \
                             SET_RX_CHECKS(honda_common_rx_checks, ret);
  }

  if (enable_gas_interceptor) {
    SET_TX_MSGS(HONDA_N_INTERCEPTOR_TX_MSGS, ret);
  } else {
    SET_TX_MSGS(HONDA_N_TX_MSGS, ret);
  }
  return ret;
}

static safety_config honda_bosch_init(uint16_t param) {
  honda_hw = HONDA_BOSCH;
  honda_brake_switch_prev = false;
  honda_bosch_radarless = GET_FLAG(param, HONDA_PARAM_RADARLESS);
  // Checking for alternate brake override from safety parameter
  honda_alt_brake_msg = GET_FLAG(param, HONDA_PARAM_ALT_BRAKE);

  // radar disabled so allow gas/brakes
#ifdef ALLOW_DEBUG
  honda_bosch_long = GET_FLAG(param, HONDA_PARAM_BOSCH_LONG);
#endif

  safety_config ret;
  if (honda_bosch_radarless && honda_alt_brake_msg) {
    SET_RX_CHECKS(honda_common_alt_brake_rx_checks, ret);
  } else if (honda_bosch_radarless) {
    SET_RX_CHECKS(honda_common_rx_checks, ret);
  } else if (honda_alt_brake_msg) {
    SET_RX_CHECKS(honda_bosch_alt_brake_rx_checks, ret);
  } else {
    SET_RX_CHECKS(honda_bosch_rx_checks, ret);
  }

  if (honda_bosch_radarless) {
    honda_bosch_long ? SET_TX_MSGS(HONDA_RADARLESS_LONG_TX_MSGS, ret) : \
                       SET_TX_MSGS(HONDA_RADARLESS_TX_MSGS, ret);
  } else {
    honda_bosch_long ? SET_TX_MSGS(HONDA_BOSCH_LONG_TX_MSGS, ret) : \
                       SET_TX_MSGS(HONDA_BOSCH_TX_MSGS, ret);
  }
  return ret;
}

static int honda_nidec_fwd_hook(int bus_num, int addr) {
  // fwd from car to camera. also fwd certain msgs from camera to car
  // 0xE4 is steering on all cars except CRV and RDX, 0x194 for CRV and RDX,
  // 0x1FA is brake control, 0x30C is acc hud, 0x33D is lkas hud
  int bus_fwd = -1;

  if (bus_num == 0) {
    bus_fwd = 2;
  }

  if (bus_num == 2) {
    // block stock lkas messages and stock acc messages (if OP is doing ACC)
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

static int honda_bosch_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;

  if (bus_num == 0) {
    bus_fwd = 2;
  }
  if (bus_num == 2)  {
    bool is_lkas_msg = (addr == 0xE4) || (addr == 0xE5) || (addr == 0x33D) || (addr == 0x33DA) || (addr == 0x33DB);
    bool is_acc_msg = ((addr == 0x1C8) || (addr == 0x30C)) && honda_bosch_radarless && honda_bosch_long;
    bool block_msg = is_lkas_msg || is_acc_msg;
    if (!block_msg) {
      bus_fwd = 0;
    }
  }

  return bus_fwd;
}

const safety_hooks honda_nidec_hooks = {
  .init = honda_nidec_init,
  .rx = honda_rx_hook,
  .tx = honda_tx_hook,
  .fwd = honda_nidec_fwd_hook,
  .get_counter = honda_get_counter,
  .get_checksum = honda_get_checksum,
  .compute_checksum = honda_compute_checksum,
};

const safety_hooks honda_bosch_hooks = {
  .init = honda_bosch_init,
  .rx = honda_rx_hook,
  .tx = honda_tx_hook,
  .fwd = honda_bosch_fwd_hook,
  .get_counter = honda_get_counter,
  .get_checksum = honda_get_checksum,
  .compute_checksum = honda_compute_checksum,
};
