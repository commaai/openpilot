#ifndef SAFETY_MODES_HYUNDAI_H
#define SAFETY_MODES_HYUNDAI_H

#include "opendbc/safety/declarations.h"
#include "opendbc/safety/modes/hyundai_common.h"

#define HYUNDAI_LIMITS(steer, rate_up, rate_down) { \
  .max_torque = (steer), \
  .max_rate_up = (rate_up), \
  .max_rate_down = (rate_down), \
  .max_rt_delta = 112, \
  .driver_torque_allowance = 50, \
  .driver_torque_multiplier = 2, \
  .type = TorqueDriverLimited, \
   /* the EPS faults when the steering angle is above a certain threshold for too long. to prevent this, */ \
   /* we allow setting CF_Lkas_ActToi bit to 0 while maintaining the requested torque value for two consecutive frames */ \
  .min_valid_request_frames = 89, \
  .max_invalid_request_frames = 2, \
  .min_valid_request_rt_interval = 810000,  /* 810ms; a ~10% buffer on cutting every 90 frames */ \
  .has_steer_req_tolerance = true, \
}

static const LongitudinalLimits HYUNDAI_LONG_LIMITS = {
  .max_accel = 200,   // 1/100 m/s2
  .min_accel = -350,  // 1/100 m/s2
};

#define HYUNDAI_COMMON_TX_MSGS(scc_bus) \
  {0x340, 0,       8, .check_relay = true},   /* LKAS11 Bus 0                              */ \
  {0x4F1, (scc_bus), 4, .check_relay = false},  /* CLU11 Bus 0 (radar-SCC) or 2 (camera-SCC) */ \
  {0x485, 0,       4, .check_relay = true},   /* LFAHDA_MFC Bus 0                          */ \

#define HYUNDAI_LONG_COMMON_TX_MSGS(scc_bus) \
  HYUNDAI_COMMON_TX_MSGS(scc_bus) \
  {0x420, 0,       8, .check_relay = true},   /* SCC11 Bus 0       */ \
  {0x421, 0,       8, .check_relay = true},   /* SCC12 Bus 0       */ \
  {0x50A, 0,       8, .check_relay = true},   /* SCC13 Bus 0       */ \
  {0x389, 0,       8, .check_relay = true},   /* SCC14 Bus 0       */ \
  {0x4A2, 0,       2, .check_relay = false},  /* FRT_RADAR11 Bus 0 */ \

#define HYUNDAI_COMMON_RX_CHECKS                                                                                                       \
  {.msg = {{0x260, 0, 8, 100U, .max_counter = 3U, .ignore_quality_flag = true},                                                         \
           {0x371, 0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }}},                  \
  {.msg = {{0x251, 0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},            \
  {.msg = {{0x4F1, 0, 4, 50U, .ignore_checksum = true, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},                \

// WHL_SPD11 and TCS13
#define HYUNDAI_WHL_SPD_TCS_RX_CHECKS                                                                                                  \
  {.msg = {{0x386, 0, 8, 100U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},                                        \
  {.msg = {{0x394, 0, 8, 100U, .max_counter = 7U, .ignore_quality_flag = true}, { 0 }, { 0 }}},                                         \

// older hyundai models have no counters and checksums on WHL_SPD11 and TCS13
#define HYUNDAI_LEGACY_WHL_SPD_TCS_RX_CHECKS                                                                                           \
  {.msg = {{0x386, 0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},           \
  {.msg = {{0x394, 0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},           \

#define HYUNDAI_SCC12_ADDR_CHECK(scc_bus)                                                                            \
  {.msg = {{0x421, (scc_bus), 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \

#define HYUNDAI_FCEV_GAS_ADDR_CHECK \
  {.msg = {{0x91,  0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \

static const CanMsg HYUNDAI_TX_MSGS[] = {
  HYUNDAI_COMMON_TX_MSGS(0)
};

static uint8_t hyundai_get_counter(const CANPacket_t *msg) {
  uint8_t cnt = 0;
  if (msg->addr == 0x260U) {
    cnt = (msg->data[7] >> 4) & 0x3U;
  }
  if (msg->addr == 0x386U) {
    cnt = ((msg->data[3] >> 6) << 2) | (msg->data[1] >> 6);
  }
  if (msg->addr == 0x394U) {
    cnt = (msg->data[1] >> 5) & 0x7U;
  }
  if (msg->addr == 0x421U) {
    cnt = msg->data[7] & 0xFU;
  }
  if (msg->addr == 0x4F1U) {
    cnt = (msg->data[3] >> 4) & 0xFU;
  }
  return cnt;
}

static uint32_t hyundai_get_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  if (msg->addr == 0x260U) {
    chksum = msg->data[7] & 0xFU;
  }
  if (msg->addr == 0x386U) {
    chksum = ((msg->data[7] >> 6) << 2) | (msg->data[5] >> 6);
  }
  if (msg->addr == 0x394U) {
    chksum = msg->data[6] & 0xFU;
  }
  if (msg->addr == 0x421U) {
    chksum = msg->data[7] >> 4;
  }
  return chksum;
}

static uint32_t hyundai_compute_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  if (msg->addr == 0x386U) {
    // count the bits
    for (int i = 0; i < 8; i++) {
      uint8_t b = msg->data[i];
      // exclude checksum and counter, the top two bits of odd bytes
      const int bit_count = ((i % 2) == 0) ? 8 : 6;
      for (int j = 0; j < bit_count; j++) {
        chksum += (b >> (uint8_t)j) & 1U;
      }
    }
    chksum = (chksum ^ 9U) & 15U;
  } else {
    // sum of nibbles
    for (int i = 0; i < 8; i++) {
      if ((msg->addr == 0x394U) && (i == 7)) {
        continue;  // exclude
      }
      uint8_t b = msg->data[i];
      if (((msg->addr == 0x260U) && (i == 7)) || ((msg->addr == 0x394U) && (i == 6)) || ((msg->addr == 0x421U) && (i == 7))) {
        b &= (msg->addr == 0x421U) ? 0x0FU : 0xF0U;  // remove checksum
      }
      chksum += (b % 16U) + (b / 16U);
    }
    chksum = (16U - (chksum %  16U)) % 16U;
  }

  return chksum;
}

static void hyundai_rx_hook(const CANPacket_t *msg) {
  // SCC12 is on bus 2 for camera-based SCC cars, bus 0 on all others
  if (!safety_setup.hyundai_longitudinal && msg_matches(msg, 0x421U, safety_setup.hyundai_camera_scc ? 2U : 0U)) {
    // 2 bits: 13-14
    bool cruise_engaged = ((GET_BYTES_LE(msg, 0, 4) >> 13) & 0x3U) != 0U;
    hyundai_common_cruise_state_check(cruise_engaged);
  }

  if (msg_matches(msg, 0x251U, 0U)) {
    int torque_driver_new = from_offset(GET_BYTES_LE(msg, 0, 2) & 0x7ffU, 1024);
    // update array of samples
    update_sample(&safety.torque_driver, torque_driver_new);
  }

  // ACC steering wheel buttons
  if (msg_matches(msg, 0x4F1U, 0U)) {
    uint8_t cruise_button = msg->data[0] & 0x7U;
    bool main_button = GET_BIT(msg, 3U);
    hyundai_common_cruise_buttons_check(cruise_button, main_button);
  }

  // gas press, different for EV, hybrid, and ICE models
  if (msg_matches(msg, 0x371U, 0U) && safety_setup.hyundai_ev_gas_signal) {
    // Accel_Pedal_Pos: bit 7 of byte 3 and bits 0-6 of byte 4
    safety.gas_pressed = ((msg->data[3] >> 7) | (msg->data[4] & 0x7FU)) != 0U;
  } else if (msg_matches(msg, 0x371U, 0U) && safety_setup.hyundai_hybrid_gas_signal) {
    safety.gas_pressed = msg->data[7] != 0U;
  } else if (safety_setup.hyundai_fcev_gas_signal && msg_matches(msg, 0x91U, 0U)) {
    safety.gas_pressed = msg->data[6] != 0U;
  } else if (msg_matches(msg, 0x260U, 0U) && !safety_setup.hyundai_ev_gas_signal && !safety_setup.hyundai_hybrid_gas_signal) {
    safety.gas_pressed = (msg->data[7] >> 6) != 0U;
  } else {
    // not a gas message for this platform
  }

  // sample wheel speed, averaging opposite corners
  if (msg_matches(msg, 0x386U, 0U)) {
    uint32_t front_left_speed = GET_BYTES_LE(msg, 0, 2) & 0x3FFFU;
    uint32_t rear_right_speed = GET_BYTES_LE(msg, 6, 2) & 0x3FFFU;
    safety.vehicle_moving = (front_left_speed > HYUNDAI_STANDSTILL_THRSLD) || (rear_right_speed > HYUNDAI_STANDSTILL_THRSLD);
  }

  if (msg_matches(msg, 0x394U, 0U)) {
    safety.brake_pressed = ((msg->data[5] >> 5U) & 0x3U) == 0x2U;
  }
}

static bool hyundai_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits HYUNDAI_STEERING_LIMITS = HYUNDAI_LIMITS(384, 3, 7);
  const TorqueSteeringLimits HYUNDAI_STEERING_LIMITS_ALT = HYUNDAI_LIMITS(270, 2, 3);
  const TorqueSteeringLimits HYUNDAI_STEERING_LIMITS_ALT_2 = HYUNDAI_LIMITS(170, 2, 3);

  bool tx = true;

  // FCA11: Block any potential actuation
  if (msg->addr == 0x38DU) {
    uint8_t CR_VSM_DecCmd = msg->data[1];
    bool FCA_CmdAct = GET_BIT(msg, 20U);
    bool CF_VSM_DecCmdAct = GET_BIT(msg, 31U);

    if ((CR_VSM_DecCmd != 0U) || FCA_CmdAct || CF_VSM_DecCmdAct) {
      tx = false;
    }
  }

  // ACCEL: safety check
  if (msg->addr == 0x421U) {
    int desired_accel_raw = from_offset(GET_BYTES_LE(msg, 3, 2) & 0x7FFU, 1023);
    int desired_accel_val = from_offset(GET_BYTES_LE(msg, 4, 2) >> 5, 1023);

    uint8_t aeb_decel_cmd = msg->data[2];
    bool aeb_req = GET_BIT(msg, 54U);
    bool aeb_stop_req = GET_BIT(msg, 55U);

    const bool raw_violation = longitudinal_accel_checks(desired_accel_raw, HYUNDAI_LONG_LIMITS);
    const bool val_violation = longitudinal_accel_checks(desired_accel_val, HYUNDAI_LONG_LIMITS);
    bool violation = raw_violation || val_violation;
    if ((aeb_decel_cmd != 0U) || aeb_req || aeb_stop_req) {
      violation = true;
    }

    if (violation) {
      tx = false;
    }
  }

  // LKA STEER: safety check
  if (msg->addr == 0x340U) {
    int desired_torque = from_offset((GET_BYTES_LE(msg, 0, 4) >> 16) & 0x7ffU, 1024);
    bool steer_req = GET_BIT(msg, 27U);

    const TorqueSteeringLimits limits = safety_setup.hyundai_alt_limits_2 ? HYUNDAI_STEERING_LIMITS_ALT_2 :
                                        safety_setup.hyundai_alt_limits ? HYUNDAI_STEERING_LIMITS_ALT : HYUNDAI_STEERING_LIMITS;

    if (steer_torque_cmd_checks(desired_torque, steer_req, limits)) {
      tx = false;
    }
  }

  // UDS: Only tester present ("\x02\x3E\x80\x00\x00\x00\x00\x00") allowed on diagnostics address
  if (msg->addr == 0x7D0U) {
    if (GET_BYTES_64_LE(msg, 0, 8) != 0x0000000000803E02ULL) {
      tx = false;
    }
  }

  // BUTTONS: used for resume spamming and cruise cancellation
  if ((msg->addr == 0x4F1U) && !safety_setup.hyundai_longitudinal) {
    uint8_t button = msg->data[0] & 0x7U;

    bool allowed_resume = (button == 1U) && safety.controls_allowed;
    bool allowed_cancel = (button == 4U) && safety.cruise_engaged_prev;
    if (!(allowed_resume || allowed_cancel)) {
      tx = false;
    }
  }

  return tx;
}

static safety_config hyundai_init(uint16_t param) {
  static const CanMsg HYUNDAI_LONG_TX_MSGS[] = {
    HYUNDAI_LONG_COMMON_TX_MSGS(0)
    {0x38D, 0, 8, .check_relay = false},  // FCA11 Bus 0
    {0x483, 0, 8, .check_relay = false},  // FCA12 Bus 0
    {0x7D0, 0, 8, .check_relay = false},  // radar UDS TX addr Bus 0 (for radar disable)
  };

  static const CanMsg HYUNDAI_CAMERA_SCC_TX_MSGS[] = {
    HYUNDAI_COMMON_TX_MSGS(2)
  };

  static const CanMsg HYUNDAI_CAMERA_SCC_LONG_TX_MSGS[] = {
    HYUNDAI_LONG_COMMON_TX_MSGS(2)
  };

  hyundai_common_init(param);
  safety_setup.hyundai_legacy = false;

  safety_config ret;
  if (safety_setup.hyundai_longitudinal) {
    // Use CLU11 (buttons) to manage controls allowed instead of SCC cruise state
    static const RxCheck hyundai_long_rx_checks[] = {
      HYUNDAI_COMMON_RX_CHECKS
      HYUNDAI_WHL_SPD_TCS_RX_CHECKS
    };

    static const RxCheck hyundai_fcev_long_rx_checks[] = {
      HYUNDAI_COMMON_RX_CHECKS
      HYUNDAI_WHL_SPD_TCS_RX_CHECKS
      HYUNDAI_FCEV_GAS_ADDR_CHECK
    };

    if (safety_setup.hyundai_fcev_gas_signal) {
      SET_RX_CHECKS(hyundai_fcev_long_rx_checks, ret);
    } else {
      SET_RX_CHECKS(hyundai_long_rx_checks, ret);
    }
    if (safety_setup.hyundai_camera_scc) {
      SET_TX_MSGS(HYUNDAI_CAMERA_SCC_LONG_TX_MSGS, ret);
    } else {
      SET_TX_MSGS(HYUNDAI_LONG_TX_MSGS, ret);
    }
  } else if (safety_setup.hyundai_camera_scc) {
    static const RxCheck hyundai_cam_scc_rx_checks[] = {
      HYUNDAI_COMMON_RX_CHECKS
      HYUNDAI_WHL_SPD_TCS_RX_CHECKS
      HYUNDAI_SCC12_ADDR_CHECK(2)
    };

    ret = BUILD_SAFETY_CFG(hyundai_cam_scc_rx_checks, HYUNDAI_CAMERA_SCC_TX_MSGS);
  } else {
    static const RxCheck hyundai_rx_checks[] = {
       HYUNDAI_COMMON_RX_CHECKS
       HYUNDAI_WHL_SPD_TCS_RX_CHECKS
       HYUNDAI_SCC12_ADDR_CHECK(0)
    };

    static const RxCheck hyundai_fcev_rx_checks[] = {
      HYUNDAI_COMMON_RX_CHECKS
      HYUNDAI_WHL_SPD_TCS_RX_CHECKS
      HYUNDAI_SCC12_ADDR_CHECK(0)
      HYUNDAI_FCEV_GAS_ADDR_CHECK
    };

    SET_TX_MSGS(HYUNDAI_TX_MSGS, ret);
    if (safety_setup.hyundai_fcev_gas_signal) {
      SET_RX_CHECKS(hyundai_fcev_rx_checks, ret);
    } else {
      SET_RX_CHECKS(hyundai_rx_checks, ret);
    }
  }
  return ret;
}

static safety_config hyundai_legacy_init(uint16_t param) {
  // older hyundai models have less checks due to missing counters and checksums
  static const RxCheck hyundai_legacy_rx_checks[] = {
    HYUNDAI_COMMON_RX_CHECKS
    HYUNDAI_LEGACY_WHL_SPD_TCS_RX_CHECKS
    HYUNDAI_SCC12_ADDR_CHECK(0)
  };

  hyundai_common_init(param);
  safety_setup.hyundai_legacy = true;
  safety_setup.hyundai_longitudinal = false;
  safety_setup.hyundai_camera_scc = false;
  return BUILD_SAFETY_CFG(hyundai_legacy_rx_checks, HYUNDAI_TX_MSGS);
}

static const safety_hooks hyundai_hooks = {
  .init = hyundai_init,
  .rx = hyundai_rx_hook,
  .tx = hyundai_tx_hook,
  .get_counter = hyundai_get_counter,
  .get_checksum = hyundai_get_checksum,
  .compute_checksum = hyundai_compute_checksum,
};

static const safety_hooks hyundai_legacy_hooks = {
  .init = hyundai_legacy_init,
  .rx = hyundai_rx_hook,
  .tx = hyundai_tx_hook,
  .get_counter = hyundai_get_counter,
  .get_checksum = hyundai_get_checksum,
  .compute_checksum = hyundai_compute_checksum,
};

#endif  // SAFETY_MODES_HYUNDAI_H
