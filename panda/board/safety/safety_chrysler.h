const SteeringLimits CHRYSLER_STEERING_LIMITS = {
  .max_steer = 261,
  .max_rt_delta = 112,
  .max_rt_interval = 250000,
  .max_rate_up = 3,
  .max_rate_down = 3,
  .max_torque_error = 80,
  .type = TorqueMotorLimited,
};

const SteeringLimits CHRYSLER_RAM_DT_STEERING_LIMITS = {
  .max_steer = 350,
  .max_rt_delta = 112,
  .max_rt_interval = 250000,
  .max_rate_up = 6,
  .max_rate_down = 6,
  .max_torque_error = 80,
  .type = TorqueMotorLimited,
};

const SteeringLimits CHRYSLER_RAM_HD_STEERING_LIMITS = {
  .max_steer = 361,
  .max_rt_delta = 182,
  .max_rt_interval = 250000,
  .max_rate_up = 14,
  .max_rate_down = 14,
  .max_torque_error = 80,
  .type = TorqueMotorLimited,
};

typedef struct {
  const int EPS_2;
  const int ESP_1;
  const int ESP_8;
  const int ECM_5;
  const int DAS_3;
  const int DAS_6;
  const int LKAS_COMMAND;
  const int CRUISE_BUTTONS;
} ChryslerAddrs;

// CAN messages for Chrysler/Jeep platforms
const ChryslerAddrs CHRYSLER_ADDRS = {
  .EPS_2            = 544,  // EPS driver input torque
  .ESP_1            = 320,  // Brake pedal and vehicle speed
  .ESP_8            = 284,  // Brake pedal and vehicle speed
  .ECM_5            = 559,  // Throttle position sensor
  .DAS_3            = 500,  // ACC engagement states from DASM
  .DAS_6            = 678,  // LKAS HUD and auto headlight control from DASM
  .LKAS_COMMAND     = 658,  // LKAS controls from DASM
  .CRUISE_BUTTONS   = 571,  // Cruise control buttons
};

// CAN messages for the 5th gen RAM DT platform
const ChryslerAddrs CHRYSLER_RAM_DT_ADDRS = {
  .EPS_2            = 49,   // EPS driver input torque
  .ESP_1            = 131,  // Brake pedal and vehicle speed
  .ESP_8            = 121,  // Brake pedal and vehicle speed
  .ECM_5            = 157,  // Throttle position sensor
  .DAS_3            = 153,  // ACC engagement states from DASM
  .DAS_6            = 250,  // LKAS HUD and auto headlight control from DASM
  .LKAS_COMMAND     = 166,  // LKAS controls from DASM
  .CRUISE_BUTTONS   = 177,  // Cruise control buttons
};

// CAN messages for the 5th gen RAM HD platform
const ChryslerAddrs CHRYSLER_RAM_HD_ADDRS = {
  .EPS_2            = 544,  // EPS driver input torque
  .ESP_1            = 320,  // Brake pedal and vehicle speed
  .ESP_8            = 284,  // Brake pedal and vehicle speed
  .ECM_5            = 559,  // Throttle position sensor
  .DAS_3            = 500,  // ACC engagement states from DASM
  .DAS_6            = 629,  // LKAS HUD and auto headlight control from DASM
  .LKAS_COMMAND     = 630,  // LKAS controls from DASM
  .CRUISE_BUTTONS   = 570,  // Cruise control buttons
};

const CanMsg CHRYSLER_TX_MSGS[] = {
  {CHRYSLER_ADDRS.CRUISE_BUTTONS, 0, 3},
  {CHRYSLER_ADDRS.LKAS_COMMAND, 0, 6},
  {CHRYSLER_ADDRS.DAS_6, 0, 8},
};

const CanMsg CHRYSLER_RAM_DT_TX_MSGS[] = {
  {CHRYSLER_RAM_DT_ADDRS.CRUISE_BUTTONS, 2, 3},
  {CHRYSLER_RAM_DT_ADDRS.LKAS_COMMAND, 0, 8},
  {CHRYSLER_RAM_DT_ADDRS.DAS_6, 0, 8},
};

const CanMsg CHRYSLER_RAM_HD_TX_MSGS[] = {
  {CHRYSLER_RAM_HD_ADDRS.CRUISE_BUTTONS, 2, 3},
  {CHRYSLER_RAM_HD_ADDRS.LKAS_COMMAND, 0, 8},
  {CHRYSLER_RAM_HD_ADDRS.DAS_6, 0, 8},
};

AddrCheckStruct chrysler_addr_checks[] = {
  {.msg = {{CHRYSLER_ADDRS.EPS_2, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_ADDRS.ESP_1, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  //{.msg = {{ESP_8, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}}},
  {.msg = {{514, 0, 8, .check_checksum = false, .max_counter = 0U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_ADDRS.ECM_5, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_ADDRS.DAS_3, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
};
#define CHRYSLER_ADDR_CHECK_LEN (sizeof(chrysler_addr_checks) / sizeof(chrysler_addr_checks[0]))

AddrCheckStruct chrysler_ram_dt_addr_checks[] = {
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.EPS_2, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.ESP_1, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.ESP_8, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.ECM_5, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.DAS_3, 2, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
};
#define CHRYSLER_RAM_DT_ADDR_CHECK_LEN (sizeof(chrysler_ram_dt_addr_checks) / sizeof(chrysler_ram_dt_addr_checks[0]))

AddrCheckStruct chrysler_ram_hd_addr_checks[] = {
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.EPS_2, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.ESP_1, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.ESP_8, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.ECM_5, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.DAS_3, 2, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
};
#define CHRYSLER_RAM_HD_ADDR_CHECK_LEN (sizeof(chrysler_ram_hd_addr_checks) / sizeof(chrysler_ram_hd_addr_checks[0]))


addr_checks chrysler_rx_checks = {chrysler_addr_checks, CHRYSLER_ADDR_CHECK_LEN};

const uint32_t CHRYSLER_PARAM_RAM_DT = 1U;  // set for Ram DT platform
const uint32_t CHRYSLER_PARAM_RAM_HD = 2U;  // set for Ram DT platform

enum {
  CHRYSLER_RAM_DT,
  CHRYSLER_RAM_HD,
  CHRYSLER_PACIFICA,  // plus Jeep
} chrysler_platform = CHRYSLER_PACIFICA;
const ChryslerAddrs *chrysler_addrs = &CHRYSLER_ADDRS;

static uint32_t chrysler_get_checksum(CANPacket_t *to_push) {
  int checksum_byte = GET_LEN(to_push) - 1U;
  return (uint8_t)(GET_BYTE(to_push, checksum_byte));
}

static uint32_t chrysler_compute_checksum(CANPacket_t *to_push) {
  // TODO: clean this up
  // http://illmatics.com/Remote%20Car%20Hacking.pdf
  uint8_t checksum = 0xFFU;
  int len = GET_LEN(to_push);
  for (int j = 0; j < (len - 1); j++) {
    uint8_t shift = 0x80U;
    uint8_t curr = (uint8_t)GET_BYTE(to_push, j);
    for (int i=0; i<8; i++) {
      uint8_t bit_sum = curr & shift;
      uint8_t temp_chk = checksum & 0x80U;
      if (bit_sum != 0U) {
        bit_sum = 0x1C;
        if (temp_chk != 0U) {
          bit_sum = 1;
        }
        checksum = checksum << 1;
        temp_chk = checksum | 1U;
        bit_sum ^= temp_chk;
      } else {
        if (temp_chk != 0U) {
          bit_sum = 0x1D;
        }
        checksum = checksum << 1;
        bit_sum ^= checksum;
      }
      checksum = bit_sum;
      shift = shift >> 1;
    }
  }
  return (uint8_t)(~checksum);
}

static uint8_t chrysler_get_counter(CANPacket_t *to_push) {
  return (uint8_t)(GET_BYTE(to_push, 6) >> 4);
}

static int chrysler_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &chrysler_rx_checks,
                                 chrysler_get_checksum, chrysler_compute_checksum,
                                 chrysler_get_counter, NULL);

  const int bus = GET_BUS(to_push);
  const int addr = GET_ADDR(to_push);

  if (valid) {

    // Measured EPS torque
    if ((bus == 0) && (addr == chrysler_addrs->EPS_2)) {
      int torque_meas_new = ((GET_BYTE(to_push, 4) & 0x7U) << 8) + GET_BYTE(to_push, 5) - 1024U;
      update_sample(&torque_meas, torque_meas_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    const int das_3_bus = (chrysler_platform == CHRYSLER_PACIFICA) ? 0 : 2;
    if ((bus == das_3_bus) && (addr == chrysler_addrs->DAS_3)) {
      bool cruise_engaged = GET_BIT(to_push, 21U) == 1U;
      pcm_cruise_check(cruise_engaged);
    }

    // TODO: use the same message for both
    // update vehicle moving
    if ((chrysler_platform != CHRYSLER_PACIFICA) && (bus == 0) && (addr == chrysler_addrs->ESP_8)) {
      vehicle_moving = ((GET_BYTE(to_push, 4) << 8) + GET_BYTE(to_push, 5)) != 0U;
    }
    if ((chrysler_platform == CHRYSLER_PACIFICA) && (bus == 0) && (addr == 514)) {
      int speed_l = (GET_BYTE(to_push, 0) << 4) + (GET_BYTE(to_push, 1) >> 4);
      int speed_r = (GET_BYTE(to_push, 2) << 4) + (GET_BYTE(to_push, 3) >> 4);
      vehicle_moving = (speed_l != 0) || (speed_r != 0);
    }

    // exit controls on rising edge of gas press
    if ((bus == 0) && (addr == chrysler_addrs->ECM_5)) {
      gas_pressed = GET_BYTE(to_push, 0U) != 0U;
    }

    // exit controls on rising edge of brake press
    if ((bus == 0) && (addr == chrysler_addrs->ESP_1)) {
      brake_pressed = ((GET_BYTE(to_push, 0U) & 0xFU) >> 2U) == 1U;
    }

    generic_rx_checks((bus == 0) && (addr == chrysler_addrs->LKAS_COMMAND));
  }
  return valid;
}

static int chrysler_tx_hook(CANPacket_t *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);

  if (chrysler_platform == CHRYSLER_RAM_DT) {
    tx = msg_allowed(to_send, CHRYSLER_RAM_DT_TX_MSGS, sizeof(CHRYSLER_RAM_DT_TX_MSGS) / sizeof(CHRYSLER_RAM_DT_TX_MSGS[0]));
  } else if (chrysler_platform == CHRYSLER_RAM_HD) {
    tx = msg_allowed(to_send, CHRYSLER_RAM_HD_TX_MSGS, sizeof(CHRYSLER_RAM_HD_TX_MSGS) / sizeof(CHRYSLER_RAM_HD_TX_MSGS[0]));
  } else {
    tx = msg_allowed(to_send, CHRYSLER_TX_MSGS, sizeof(CHRYSLER_TX_MSGS) / sizeof(CHRYSLER_TX_MSGS[0]));
  }

  // STEERING
  if (tx && (addr == chrysler_addrs->LKAS_COMMAND)) {
    int start_byte = (chrysler_platform == CHRYSLER_PACIFICA) ? 0 : 1;
    int desired_torque = ((GET_BYTE(to_send, start_byte) & 0x7U) << 8) | GET_BYTE(to_send, start_byte + 1);
    desired_torque -= 1024;

    const SteeringLimits limits = (chrysler_platform == CHRYSLER_PACIFICA) ? CHRYSLER_STEERING_LIMITS :
                                  (chrysler_platform == CHRYSLER_RAM_DT) ? CHRYSLER_RAM_DT_STEERING_LIMITS : CHRYSLER_RAM_HD_STEERING_LIMITS;
    if (steer_torque_cmd_checks(desired_torque, -1, limits)) {
      tx = 0;
    }
  }

  // FORCE CANCEL: only the cancel button press is allowed
  if (addr == chrysler_addrs->CRUISE_BUTTONS) {
    const bool is_cancel = GET_BYTE(to_send, 0) == 1U;
    const bool is_resume = GET_BYTE(to_send, 0) == 0x10U;
    const bool allowed = is_cancel || (is_resume && controls_allowed);
    if (!allowed) {
      tx = 0;
    }
  }

  return tx;
}

static int chrysler_fwd_hook(int bus_num, CANPacket_t *to_fwd) {
  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);

  // forward to camera
  if (bus_num == 0) {
    bus_fwd = 2;
  }

  // forward all messages from camera except LKAS messages
  const bool is_lkas = ((addr == chrysler_addrs->LKAS_COMMAND) || (addr == chrysler_addrs->DAS_6));
  if ((bus_num == 2) && !is_lkas){
    bus_fwd = 0;
  }

  return bus_fwd;
}

static const addr_checks* chrysler_init(uint16_t param) {
  if (GET_FLAG(param, CHRYSLER_PARAM_RAM_DT)) {
    chrysler_platform = CHRYSLER_RAM_DT;
    chrysler_addrs = &CHRYSLER_RAM_DT_ADDRS;
    chrysler_rx_checks = (addr_checks){chrysler_ram_dt_addr_checks, CHRYSLER_RAM_DT_ADDR_CHECK_LEN};
  } else if (GET_FLAG(param, CHRYSLER_PARAM_RAM_HD)) {
#ifdef ALLOW_DEBUG
    chrysler_platform = CHRYSLER_RAM_HD;
    chrysler_addrs = &CHRYSLER_RAM_HD_ADDRS;
    chrysler_rx_checks = (addr_checks){chrysler_ram_hd_addr_checks, CHRYSLER_RAM_HD_ADDR_CHECK_LEN};
#endif
  } else {
    chrysler_platform = CHRYSLER_PACIFICA;
    chrysler_addrs = &CHRYSLER_ADDRS;
    chrysler_rx_checks = (addr_checks){chrysler_addr_checks, CHRYSLER_ADDR_CHECK_LEN};
  }

  return &chrysler_rx_checks;
}

const safety_hooks chrysler_hooks = {
  .init = chrysler_init,
  .rx = chrysler_rx_hook,
  .tx = chrysler_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = chrysler_fwd_hook,
};
