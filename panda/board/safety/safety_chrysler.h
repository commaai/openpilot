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
  .EPS_2            = 0x220,  // EPS driver input torque
  .ESP_1            = 0x140,  // Brake pedal and vehicle speed
  .ESP_8            = 0x11C,  // Brake pedal and vehicle speed
  .ECM_5            = 0x22F,  // Throttle position sensor
  .DAS_3            = 0x1F4,  // ACC engagement states from DASM
  .DAS_6            = 0x2A6,  // LKAS HUD and auto headlight control from DASM
  .LKAS_COMMAND     = 0x292,  // LKAS controls from DASM
  .CRUISE_BUTTONS   = 0x23B,  // Cruise control buttons
};

// CAN messages for the 5th gen RAM DT platform
const ChryslerAddrs CHRYSLER_RAM_DT_ADDRS = {
  .EPS_2            = 0x31,   // EPS driver input torque
  .ESP_1            = 0x83,   // Brake pedal and vehicle speed
  .ESP_8            = 0x79,   // Brake pedal and vehicle speed
  .ECM_5            = 0x9D,   // Throttle position sensor
  .DAS_3            = 0x99,   // ACC engagement states from DASM
  .DAS_6            = 0xFA,   // LKAS HUD and auto headlight control from DASM
  .LKAS_COMMAND     = 0xA6,   // LKAS controls from DASM
  .CRUISE_BUTTONS   = 0xB1,   // Cruise control buttons
};

// CAN messages for the 5th gen RAM HD platform
const ChryslerAddrs CHRYSLER_RAM_HD_ADDRS = {
  .EPS_2            = 0x220,  // EPS driver input torque
  .ESP_1            = 0x140,  // Brake pedal and vehicle speed
  .ESP_8            = 0x11C,  // Brake pedal and vehicle speed
  .ECM_5            = 0x22F,  // Throttle position sensor
  .DAS_3            = 0x1F4,  // ACC engagement states from DASM
  .DAS_6            = 0x275,  // LKAS HUD and auto headlight control from DASM
  .LKAS_COMMAND     = 0x276,  // LKAS controls from DASM
  .CRUISE_BUTTONS   = 0x23A,  // Cruise control buttons
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

RxCheck chrysler_rx_checks[] = {
  {.msg = {{CHRYSLER_ADDRS.EPS_2, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 100U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_ADDRS.ESP_1, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
  //{.msg = {{ESP_8, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}}},
  {.msg = {{514, 0, 8, .check_checksum = false, .max_counter = 0U, .frequency = 100U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_ADDRS.ECM_5, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_ADDRS.DAS_3, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
};

RxCheck chrysler_ram_dt_rx_checks[] = {
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.EPS_2, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 100U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.ESP_1, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.ESP_8, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.ECM_5, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_DT_ADDRS.DAS_3, 2, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
};

RxCheck chrysler_ram_hd_rx_checks[] = {
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.EPS_2, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 100U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.ESP_1, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.ESP_8, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.ECM_5, 0, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{CHRYSLER_RAM_HD_ADDRS.DAS_3, 2, 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
};



const uint32_t CHRYSLER_PARAM_RAM_DT = 1U;  // set for Ram DT platform
const uint32_t CHRYSLER_PARAM_RAM_HD = 2U;  // set for Ram HD platform

enum {
  CHRYSLER_RAM_DT,
  CHRYSLER_RAM_HD,
  CHRYSLER_PACIFICA,  // plus Jeep
} chrysler_platform = CHRYSLER_PACIFICA;
const ChryslerAddrs *chrysler_addrs = &CHRYSLER_ADDRS;

static uint32_t chrysler_get_checksum(const CANPacket_t *to_push) {
  int checksum_byte = GET_LEN(to_push) - 1U;
  return (uint8_t)(GET_BYTE(to_push, checksum_byte));
}

static uint32_t chrysler_compute_checksum(const CANPacket_t *to_push) {
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

static uint8_t chrysler_get_counter(const CANPacket_t *to_push) {
  return (uint8_t)(GET_BYTE(to_push, 6) >> 4);
}

static void chrysler_rx_hook(const CANPacket_t *to_push) {
  const int bus = GET_BUS(to_push);
  const int addr = GET_ADDR(to_push);

  // Measured EPS torque
  if ((bus == 0) && (addr == chrysler_addrs->EPS_2)) {
    int torque_meas_new = ((GET_BYTE(to_push, 4) & 0x7U) << 8) + GET_BYTE(to_push, 5) - 1024U;
    update_sample(&torque_meas, torque_meas_new);
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  const int das_3_bus = (chrysler_platform == CHRYSLER_PACIFICA) ? 0 : 2;
  if ((bus == das_3_bus) && (addr == chrysler_addrs->DAS_3)) {
    bool cruise_engaged = GET_BIT(to_push, 21U);
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

static bool chrysler_tx_hook(const CANPacket_t *to_send) {
  bool tx = true;
  int addr = GET_ADDR(to_send);

  // STEERING
  if (addr == chrysler_addrs->LKAS_COMMAND) {
    int start_byte = (chrysler_platform == CHRYSLER_PACIFICA) ? 0 : 1;
    int desired_torque = ((GET_BYTE(to_send, start_byte) & 0x7U) << 8) | GET_BYTE(to_send, start_byte + 1);
    desired_torque -= 1024;

    const SteeringLimits limits = (chrysler_platform == CHRYSLER_PACIFICA) ? CHRYSLER_STEERING_LIMITS :
                                  (chrysler_platform == CHRYSLER_RAM_DT) ? CHRYSLER_RAM_DT_STEERING_LIMITS : CHRYSLER_RAM_HD_STEERING_LIMITS;

    bool steer_req = (chrysler_platform == CHRYSLER_PACIFICA) ? GET_BIT(to_send, 4U) : (GET_BYTE(to_send, 3) & 0x7U) == 2U;
    if (steer_torque_cmd_checks(desired_torque, steer_req, limits)) {
      tx = false;
    }
  }

  // FORCE CANCEL: only the cancel button press is allowed
  if (addr == chrysler_addrs->CRUISE_BUTTONS) {
    const bool is_cancel = GET_BYTE(to_send, 0) == 1U;
    const bool is_resume = GET_BYTE(to_send, 0) == 0x10U;
    const bool allowed = is_cancel || (is_resume && controls_allowed);
    if (!allowed) {
      tx = false;
    }
  }

  return tx;
}

static int chrysler_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;

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

static safety_config chrysler_init(uint16_t param) {
  safety_config ret;

  bool enable_ram_dt = GET_FLAG(param, CHRYSLER_PARAM_RAM_DT);
  if (enable_ram_dt) {
    chrysler_platform = CHRYSLER_RAM_DT;
    chrysler_addrs = &CHRYSLER_RAM_DT_ADDRS;
    ret = BUILD_SAFETY_CFG(chrysler_ram_dt_rx_checks, CHRYSLER_RAM_DT_TX_MSGS);
#ifdef ALLOW_DEBUG
  } else if (GET_FLAG(param, CHRYSLER_PARAM_RAM_HD)) {
    chrysler_platform = CHRYSLER_RAM_HD;
    chrysler_addrs = &CHRYSLER_RAM_HD_ADDRS;
    ret = BUILD_SAFETY_CFG(chrysler_ram_hd_rx_checks, CHRYSLER_RAM_HD_TX_MSGS);
#endif
  } else {
    chrysler_platform = CHRYSLER_PACIFICA;
    chrysler_addrs = &CHRYSLER_ADDRS;
    ret = BUILD_SAFETY_CFG(chrysler_rx_checks, CHRYSLER_TX_MSGS);
  }
  return ret;
}

const safety_hooks chrysler_hooks = {
  .init = chrysler_init,
  .rx = chrysler_rx_hook,
  .tx = chrysler_tx_hook,
  .fwd = chrysler_fwd_hook,
  .get_counter = chrysler_get_counter,
  .get_checksum = chrysler_get_checksum,
  .compute_checksum = chrysler_compute_checksum,
};
