#pragma once

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

extern const LongitudinalLimits HYUNDAI_LONG_LIMITS;
const LongitudinalLimits HYUNDAI_LONG_LIMITS = {
  .max_accel = 200,   // 1/100 m/s2
  .min_accel = -350,  // 1/100 m/s2
};

#define HYUNDAI_COMMON_TX_MSGS(scc_bus) \
  {0x340, 0,       8, .check_relay = true},   /* LKAS11 Bus 0                              */ \
  {0x4F1, scc_bus, 4, .check_relay = false},  /* CLU11 Bus 0 (radar-SCC) or 2 (camera-SCC) */ \
  {0x485, 0,       4, .check_relay = true},   /* LFAHDA_MFC Bus 0                          */ \

#define HYUNDAI_LONG_COMMON_TX_MSGS(scc_bus) \
  HYUNDAI_COMMON_TX_MSGS(scc_bus) \
  {0x420, 0,       8, .check_relay = true},   /* SCC11 Bus 0       */ \
  {0x421, 0,       8, .check_relay = true},   /* SCC12 Bus 0       */ \
  {0x50A, 0,       8, .check_relay = true},   /* SCC13 Bus 0       */ \
  {0x389, 0,       8, .check_relay = true},   /* SCC14 Bus 0       */ \
  {0x4A2, 0,       2, .check_relay = false},  /* FRT_RADAR11 Bus 0 */ \

#define HYUNDAI_COMMON_RX_CHECKS(legacy)                                                                                                                                               \
  {.msg = {{0x260, 0, 8, 100U, .max_counter = 3U, .ignore_quality_flag = true},                                                                                           \
           {0x371, 0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }}},                                                    \
  {.msg = {{0x386, 0, 8, 100U, .ignore_checksum = (legacy), .ignore_counter = (legacy), .max_counter = (legacy) ? 0U : 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
  {.msg = {{0x394, 0, 8, 100U, .ignore_checksum = (legacy), .ignore_counter = (legacy), .max_counter = (legacy) ? 0U : 7U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \
  {.msg = {{0x251, 0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},                                              \
  {.msg = {{0x4F1, 0, 4, 50U, .ignore_checksum = true, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},                                                  \

#define HYUNDAI_SCC12_ADDR_CHECK(scc_bus)                                                                            \
  {.msg = {{0x421, (scc_bus), 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \

#define HYUNDAI_FCEV_GAS_ADDR_CHECK \
  {.msg = {{0x91,  0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \

static const CanMsg HYUNDAI_TX_MSGS[] = {
  HYUNDAI_COMMON_TX_MSGS(0)
};

static bool hyundai_legacy = false;

static uint8_t hyundai_get_counter(const CANPacket_t *msg) {

  uint8_t cnt = 0;
  if (msg->addr == 0x260U) {
    cnt = (msg->data[7] >> 4) & 0x3U;
  } else if (msg->addr == 0x386U) {
    cnt = ((msg->data[3] >> 6) << 2) | (msg->data[1] >> 6);
  } else if (msg->addr == 0x394U) {
    cnt = (msg->data[1] >> 5) & 0x7U;
  } else if (msg->addr == 0x421U) {
    cnt = msg->data[7] & 0xFU;
  } else if (msg->addr == 0x4F1U) {
    cnt = (msg->data[3] >> 4) & 0xFU;
  } else {
  }
  return cnt;
}

static uint32_t hyundai_get_checksum(const CANPacket_t *msg) {

  uint8_t chksum = 0;
  if (msg->addr == 0x260U) {
    chksum = msg->data[7] & 0xFU;
  } else if (msg->addr == 0x386U) {
    chksum = ((msg->data[7] >> 6) << 2) | (msg->data[5] >> 6);
  } else if (msg->addr == 0x394U) {
    chksum = msg->data[6] & 0xFU;
  } else if (msg->addr == 0x421U) {
    chksum = msg->data[7] >> 4;
  } else {
  }
  return chksum;
}

static uint32_t hyundai_compute_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  if (msg->addr == 0x386U) {
    // count the bits
    for (int i = 0; i < 8; i++) {
      uint8_t b = msg->data[i];
      for (int j = 0; j < 8; j++) {
        uint8_t bit = 0;
        // exclude checksum and counter
        if (((i != 1) || (j < 6)) && ((i != 3) || (j < 6)) && ((i != 5) || (j < 6)) && ((i != 7) || (j < 6))) {
          bit = (b >> (uint8_t)j) & 1U;
        }
        chksum += bit;
      }
    }
    chksum = (chksum ^ 9U) & 15U;
  } else {
    // sum of nibbles
    for (int i = 0; i < 8; i++) {
      if ((msg->addr == 0x394U) && (i == 7)) {
        continue; // exclude
      }
      uint8_t b = msg->data[i];
      if (((msg->addr == 0x260U) && (i == 7)) || ((msg->addr == 0x394U) && (i == 6)) || ((msg->addr == 0x421U) && (i == 7))) {
        b &= (msg->addr == 0x421U) ? 0x0FU : 0xF0U; // remove checksum
      }
      chksum += (b % 16U) + (b / 16U);
    }
    chksum = (16U - (chksum %  16U)) % 16U;
  }

  return chksum;
}

static void hyundai_rx_hook(const CANPacket_t *msg) {

  // SCC12 is on bus 2 for camera-based SCC cars, bus 0 on all others
  if (msg->addr == 0x421U) {
    if (((msg->bus == 0U) && !hyundai_camera_scc) || ((msg->bus == 2U) && hyundai_camera_scc)) {
      // 2 bits: 13-14
      int cruise_engaged = (GET_BYTES(msg, 0, 4) >> 13) & 0x3U;
      hyundai_common_cruise_state_check(cruise_engaged);
    }
  }

  if (msg->bus == 0U) {
    if (msg->addr == 0x251U) {
      int torque_driver_new = (GET_BYTES(msg, 0, 2) & 0x7ffU) - 1024U;
      // update array of samples
      update_sample(&torque_driver, torque_driver_new);
    }

    // ACC steering wheel buttons
    if (msg->addr == 0x4F1U) {
      int cruise_button = msg->data[0] & 0x7U;
      bool main_button = GET_BIT(msg, 3U);
      hyundai_common_cruise_buttons_check(cruise_button, main_button);
    }

    // gas press, different for EV, hybrid, and ICE models
    if ((msg->addr == 0x371U) && hyundai_ev_gas_signal) {
      gas_pressed = (((msg->data[4] & 0x7FU) << 1) | (msg->data[3] >> 7)) != 0U;
    } else if ((msg->addr == 0x371U) && hyundai_hybrid_gas_signal) {
      gas_pressed = msg->data[7] != 0U;
    } else if ((msg->addr == 0x91U) && hyundai_fcev_gas_signal) {
      gas_pressed = msg->data[6] != 0U;
    } else if ((msg->addr == 0x260U) && !hyundai_ev_gas_signal && !hyundai_hybrid_gas_signal) {
      gas_pressed = (msg->data[7] >> 6) != 0U;
    } else {
    }

    // sample wheel speed, averaging opposite corners
    if (msg->addr == 0x386U) {
      uint32_t front_left_speed = GET_BYTES(msg, 0, 2) & 0x3FFFU;
      uint32_t rear_right_speed = GET_BYTES(msg, 6, 2) & 0x3FFFU;
      vehicle_moving = (front_left_speed > HYUNDAI_STANDSTILL_THRSLD) || (rear_right_speed > HYUNDAI_STANDSTILL_THRSLD);
    }

    if (msg->addr == 0x394U) {
      brake_pressed = ((msg->data[5] >> 5U) & 0x3U) == 0x2U;
    }
  }
}

static bool hyundai_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits HYUNDAI_STEERING_LIMITS = HYUNDAI_LIMITS(384, 3, 7);
  const TorqueSteeringLimits HYUNDAI_STEERING_LIMITS_ALT = HYUNDAI_LIMITS(270, 2, 3);
  const TorqueSteeringLimits HYUNDAI_STEERING_LIMITS_ALT_2 = HYUNDAI_LIMITS(170, 2, 3);

  bool tx = true;

  // FCA11: Block any potential actuation
  if (msg->addr == 0x38DU) {
    int CR_VSM_DecCmd = msg->data[1];
    bool FCA_CmdAct = GET_BIT(msg, 20U);
    bool CF_VSM_DecCmdAct = GET_BIT(msg, 31U);

    if ((CR_VSM_DecCmd != 0) || FCA_CmdAct || CF_VSM_DecCmdAct) {
      tx = false;
    }
  }

  // ACCEL: safety check
  if (msg->addr == 0x421U) {
    int desired_accel_raw = (((msg->data[4] & 0x7U) << 8) | msg->data[3]) - 1023U;
    int desired_accel_val = ((msg->data[5] << 3) | (msg->data[4] >> 5)) - 1023U;

    int aeb_decel_cmd = msg->data[2];
    bool aeb_req = GET_BIT(msg, 54U);

    bool violation = false;

    violation |= longitudinal_accel_checks(desired_accel_raw, HYUNDAI_LONG_LIMITS);
    violation |= longitudinal_accel_checks(desired_accel_val, HYUNDAI_LONG_LIMITS);
    violation |= (aeb_decel_cmd != 0);
    violation |= aeb_req;

    if (violation) {
      tx = false;
    }
  }

  // LKA STEER: safety check
  if (msg->addr == 0x340U) {
    int desired_torque = ((GET_BYTES(msg, 0, 4) >> 16) & 0x7ffU) - 1024U;
    bool steer_req = GET_BIT(msg, 27U);

    const TorqueSteeringLimits limits = hyundai_alt_limits_2 ? HYUNDAI_STEERING_LIMITS_ALT_2 :
                                        hyundai_alt_limits ? HYUNDAI_STEERING_LIMITS_ALT : HYUNDAI_STEERING_LIMITS;

    if (steer_torque_cmd_checks(desired_torque, steer_req, limits)) {
      tx = false;
    }
  }

  // UDS: Only tester present ("\x02\x3E\x80\x00\x00\x00\x00\x00") allowed on diagnostics address
  if (msg->addr == 0x7D0U) {
    if ((GET_BYTES(msg, 0, 4) != 0x00803E02U) || (GET_BYTES(msg, 4, 4) != 0x0U)) {
      tx = false;
    }
  }

  // BUTTONS: used for resume spamming and cruise cancellation
  if ((msg->addr == 0x4F1U) && !hyundai_longitudinal) {
    int button = msg->data[0] & 0x7U;

    bool allowed_resume = (button == 1) && controls_allowed;
    bool allowed_cancel = (button == 4) && cruise_engaged_prev;
    if (!(allowed_resume || allowed_cancel)) {
      tx = false;
    }
  }

  return tx;
}

static safety_config hyundai_init(uint16_t param) {
  static const CanMsg HYUNDAI_LONG_TX_MSGS[] = {
    HYUNDAI_LONG_COMMON_TX_MSGS(0)
    {0x38D, 0, 8, .check_relay = false}, // FCA11 Bus 0
    {0x483, 0, 8, .check_relay = false}, // FCA12 Bus 0
    {0x7D0, 0, 8, .check_relay = false}, // radar UDS TX addr Bus 0 (for radar disable)
  };

  static const CanMsg HYUNDAI_CAMERA_SCC_TX_MSGS[] = {
    HYUNDAI_COMMON_TX_MSGS(2)
  };

  static const CanMsg HYUNDAI_CAMERA_SCC_LONG_TX_MSGS[] = {
    HYUNDAI_LONG_COMMON_TX_MSGS(2)
  };

  hyundai_common_init(param);
  hyundai_legacy = false;

  safety_config ret;
  if (hyundai_longitudinal) {
    // Use CLU11 (buttons) to manage controls allowed instead of SCC cruise state
    static RxCheck hyundai_long_rx_checks[] = {
      HYUNDAI_COMMON_RX_CHECKS(false)
    };

    static RxCheck hyundai_fcev_long_rx_checks[] = {
      HYUNDAI_COMMON_RX_CHECKS(false)
      HYUNDAI_FCEV_GAS_ADDR_CHECK
    };

    if (hyundai_fcev_gas_signal) {
      SET_RX_CHECKS(hyundai_fcev_long_rx_checks, ret);
    } else {
      SET_RX_CHECKS(hyundai_long_rx_checks, ret);
    }
    if (hyundai_camera_scc) {
      SET_TX_MSGS(HYUNDAI_CAMERA_SCC_LONG_TX_MSGS, ret);
    } else {
      SET_TX_MSGS(HYUNDAI_LONG_TX_MSGS, ret);
    }

  } else if (hyundai_camera_scc) {
    static RxCheck hyundai_cam_scc_rx_checks[] = {
      HYUNDAI_COMMON_RX_CHECKS(false)
      HYUNDAI_SCC12_ADDR_CHECK(2)
    };

    ret = BUILD_SAFETY_CFG(hyundai_cam_scc_rx_checks, HYUNDAI_CAMERA_SCC_TX_MSGS);
  } else {
    static RxCheck hyundai_rx_checks[] = {
       HYUNDAI_COMMON_RX_CHECKS(false)
       HYUNDAI_SCC12_ADDR_CHECK(0)
    };

    static RxCheck hyundai_fcev_rx_checks[] = {
      HYUNDAI_COMMON_RX_CHECKS(false)
      HYUNDAI_SCC12_ADDR_CHECK(0)
      HYUNDAI_FCEV_GAS_ADDR_CHECK
    };

    SET_TX_MSGS(HYUNDAI_TX_MSGS, ret);
    if (hyundai_fcev_gas_signal) {
      SET_RX_CHECKS(hyundai_fcev_rx_checks, ret);
    } else {
      SET_RX_CHECKS(hyundai_rx_checks, ret);
    }
  }
  return ret;
}

static safety_config hyundai_legacy_init(uint16_t param) {
  // older hyundai models have less checks due to missing counters and checksums
  static RxCheck hyundai_legacy_rx_checks[] = {
    HYUNDAI_COMMON_RX_CHECKS(true)
    HYUNDAI_SCC12_ADDR_CHECK(0)
  };

  hyundai_common_init(param);
  hyundai_legacy = true;
  hyundai_longitudinal = false;
  hyundai_camera_scc = false;
  return BUILD_SAFETY_CFG(hyundai_legacy_rx_checks, HYUNDAI_TX_MSGS);
}

const safety_hooks hyundai_hooks = {
  .init = hyundai_init,
  .rx = hyundai_rx_hook,
  .tx = hyundai_tx_hook,
  .get_counter = hyundai_get_counter,
  .get_checksum = hyundai_get_checksum,
  .compute_checksum = hyundai_compute_checksum,
};

const safety_hooks hyundai_legacy_hooks = {
  .init = hyundai_legacy_init,
  .rx = hyundai_rx_hook,
  .tx = hyundai_tx_hook,
  .get_counter = hyundai_get_counter,
  .get_checksum = hyundai_get_checksum,
  .compute_checksum = hyundai_compute_checksum,
};
