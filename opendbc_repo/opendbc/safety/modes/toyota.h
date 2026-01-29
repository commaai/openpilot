#pragma once

#include "opendbc/safety/declarations.h"

// Stock longitudinal
#define TOYOTA_BASE_TX_MSGS \
  {0x191, 0, 8, .check_relay = true}, {0x412, 0, 8, .check_relay = true}, {0x1D2, 0, 8, .check_relay = false},  /* LKAS + LTA + PCM cancel cmd */  \

#define TOYOTA_COMMON_TX_MSGS \
  TOYOTA_BASE_TX_MSGS \
  {0x2E4, 0, 5, .check_relay = true}, \
  {0x343, 0, 8, .check_relay = false},  /* ACC cancel cmd */  \

#define TOYOTA_COMMON_SECOC_TX_MSGS \
  TOYOTA_BASE_TX_MSGS \
  {0x2E4, 0, 8, .check_relay = true}, {0x131, 0, 8, .check_relay = true}, \
  {0x343, 0, 8, .check_relay = false},  /* ACC cancel cmd */ \

#define TOYOTA_COMMON_LONG_TX_MSGS \
  TOYOTA_COMMON_TX_MSGS \
  /* DSU bus 0 */ \
  {0x283, 0, 7, .check_relay = false}, {0x2E6, 0, 8, .check_relay = false}, {0x2E7, 0, 8, .check_relay = false}, {0x33E, 0, 7, .check_relay = false}, \
  {0x344, 0, 8, .check_relay = false}, {0x365, 0, 7, .check_relay = false}, {0x366, 0, 7, .check_relay = false}, {0x4CB, 0, 8, .check_relay = false}, \
  /* DSU bus 1 */ \
  {0x128, 1, 6, .check_relay = false}, {0x141, 1, 4, .check_relay = false}, {0x160, 1, 8, .check_relay = false}, {0x161, 1, 7, .check_relay = false}, \
  {0x470, 1, 4, .check_relay = false}, \
  /* PCS_HUD */                        \
  {0x411, 0, 8, .check_relay = false}, \
  /* radar diagnostic address */       \
  {0x750, 0, 8, .check_relay = false}, \
  /* ACC */                            \
  {0x343, 0, 8, .check_relay = true},  \

#define TOYOTA_COMMON_SECOC_LONG_TX_MSGS \
  TOYOTA_COMMON_SECOC_TX_MSGS \
  {0x343, 0, 8, .check_relay = true}, \
  {0x183, 0, 8, .check_relay = true},  /* ACC_CONTROL_2 */ \

#define TOYOTA_COMMON_RX_CHECKS(lta)                                                                                                       \
  {.msg = {{ 0xaa, 0, 8, 83U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \
  {.msg = {{0x260, 0, 8, 50U, .ignore_counter = true, .ignore_quality_flag=!(lta)}, { 0 }, { 0 }}},                           \

#define TOYOTA_RX_CHECKS(lta)                                                                                                               \
  TOYOTA_COMMON_RX_CHECKS(lta)                                                                                                              \
  {.msg = {{0x1D2, 0, 8, 33U, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},                            \
  {.msg = {{0x226, 0, 8, 40U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},  { 0 }, { 0 }}},  \

#define TOYOTA_ALT_BRAKE_RX_CHECKS(lta)                                                                                                    \
  TOYOTA_COMMON_RX_CHECKS(lta)                                                                                                             \
  {.msg = {{0x1D2, 0, 8, 33U, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},                           \
  {.msg = {{0x224, 0, 8, 40U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \

#define TOYOTA_SECOC_RX_CHECKS                                                                                                             \
  TOYOTA_COMMON_RX_CHECKS(false)                                                                                                           \
  {.msg = {{0x176, 0, 8, 32U, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},                           \
  {.msg = {{0x116, 0, 8, 42U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \
  {.msg = {{0x101, 0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \

static bool toyota_secoc = false;
static bool toyota_alt_brake = false;
static bool toyota_stock_longitudinal = false;
static bool toyota_lta = false;
static int toyota_dbc_eps_torque_factor = 100;   // conversion factor for STEER_TORQUE_EPS in %: see dbc file

static uint32_t toyota_compute_checksum(const CANPacket_t *msg) {
  int len = GET_LEN(msg);
  uint8_t checksum = (uint8_t)(msg->addr) + (uint8_t)((unsigned int)(msg->addr) >> 8U) + (uint8_t)(len);
  for (int i = 0; i < (len - 1); i++) {
    checksum += (uint8_t)msg->data[i];
  }
  return checksum;
}

static uint32_t toyota_get_checksum(const CANPacket_t *msg) {
  int checksum_byte = GET_LEN(msg) - 1U;
  return (uint8_t)(msg->data[checksum_byte]);
}

static bool toyota_get_quality_flag_valid(const CANPacket_t *msg) {
  return !GET_BIT(msg, 3U);  // STEER_ANGLE_INITIALIZING
}

static void toyota_rx_hook(const CANPacket_t *msg) {
  if (msg->bus == 0U) {

    // get eps motor torque (0.66 factor in dbc)
    if (msg->addr == 0x260U) {
      int torque_meas_new = (msg->data[5] << 8) | msg->data[6];
      torque_meas_new = to_signed(torque_meas_new, 16);

      // scale by dbc_factor
      torque_meas_new = (torque_meas_new * toyota_dbc_eps_torque_factor) / 100;

      // update array of sample
      update_sample(&torque_meas, torque_meas_new);

      // increase torque_meas by 1 to be conservative on rounding
      torque_meas.min--;
      torque_meas.max++;

      // driver torque for angle limiting
      int torque_driver_new = (msg->data[1] << 8) | msg->data[2];
      torque_driver_new = to_signed(torque_driver_new, 16);
      update_sample(&torque_driver, torque_driver_new);

      // LTA request angle should match current angle while inactive, clipped to max accepted angle.
      // note that angle can be relative to init angle on some TSS2 platforms, LTA has the same offset
      bool steer_angle_initializing = GET_BIT(msg, 3U);
      if (!steer_angle_initializing) {
        int angle_meas_new = (msg->data[3] << 8U) | msg->data[4];
        angle_meas_new = to_signed(angle_meas_new, 16);
        update_sample(&angle_meas, angle_meas_new);
      }
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    // exit controls on rising edge of gas press, if not alternative experience
    // exit controls on rising edge of brake press
    if (toyota_secoc) {
      if (msg->addr == 0x176U) {
        bool cruise_engaged = GET_BIT(msg, 5U);  // PCM_CRUISE.CRUISE_ACTIVE
        pcm_cruise_check(cruise_engaged);
      }
      if (msg->addr == 0x116U) {
        gas_pressed = msg->data[1] != 0U;  // GAS_PEDAL.GAS_PEDAL_USER
      }
      if (msg->addr == 0x101U) {
        brake_pressed = GET_BIT(msg, 3U);  // BRAKE_MODULE.BRAKE_PRESSED (toyota_rav4_prime_generated.dbc)
      }
    } else {
      if (msg->addr == 0x1D2U) {
        bool cruise_engaged = GET_BIT(msg, 5U);  // PCM_CRUISE.CRUISE_ACTIVE
        pcm_cruise_check(cruise_engaged);
        gas_pressed = !GET_BIT(msg, 4U);  // PCM_CRUISE.GAS_RELEASED
      }
      if (!toyota_alt_brake && (msg->addr == 0x226U)) {
        brake_pressed = GET_BIT(msg, 37U);  // BRAKE_MODULE.BRAKE_PRESSED (toyota_nodsu_pt_generated.dbc)
      }
      if (toyota_alt_brake && (msg->addr == 0x224U)) {
        brake_pressed = GET_BIT(msg, 5U);  // BRAKE_MODULE.BRAKE_PRESSED (toyota_new_mc_pt_generated.dbc)
      }
    }

    // sample speed
    if (msg->addr == 0xaaU) {
      int speed = 0;
      // sum 4 wheel speeds. conversion: raw * 0.01 - 67.67
      for (uint8_t i = 0U; i < 8U; i += 2U) {
        int wheel_speed = (msg->data[i] << 8U) | msg->data[(i + 1U)];
        speed += wheel_speed - 6767;
      }
      // check that all wheel speeds are at zero value
      vehicle_moving = speed != 0;

      UPDATE_VEHICLE_SPEED(speed / 4.0 * 0.01 * KPH_TO_MS);
    }
  }
}

static bool toyota_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits TOYOTA_TORQUE_STEERING_LIMITS = {
    .max_torque = 1500,
    .max_rate_up = 15,          // ramp up slow
    .max_rate_down = 25,        // ramp down fast
    .max_torque_error = 350,    // max torque cmd in excess of motor torque
    .max_rt_delta = 450,        // the real time limit is 1800/sec, a 20% buffer
    .type = TorqueMotorLimited,

    // the EPS faults when the steering angle rate is above a certain threshold for too long. to prevent this,
    // we allow setting STEER_REQUEST bit to 0 while maintaining the requested torque value for a single frame
    .min_valid_request_frames = 17,
    .max_invalid_request_frames = 1,
    .min_valid_request_rt_interval = 162000,  // 162ms; a ~10% buffer on cutting every 18 frames
    .has_steer_req_tolerance = true,
  };

  static const AngleSteeringLimits TOYOTA_ANGLE_STEERING_LIMITS = {
    // LTA angle limits
    // factor for STEER_TORQUE_SENSOR->STEER_ANGLE and STEERING_LTA->STEER_ANGLE_CMD (1 / 0.0573)
    .max_angle = 1657,  // EPS only accepts up to 94.9461
    .angle_deg_to_can = 17.452007,
    .angle_rate_up_lookup = {
      {5., 25., 25.},
      {0.3, 0.15, 0.15}
    },
    .angle_rate_down_lookup = {
      {5., 25., 25.},
      {0.36, 0.26, 0.26}
    },
  };

  const int TOYOTA_LTA_MAX_MEAS_TORQUE = 1500;
  const int TOYOTA_LTA_MAX_DRIVER_TORQUE = 150;

  // longitudinal limits
  const LongitudinalLimits TOYOTA_LONG_LIMITS = {
    .max_accel = 2000,   // 2.0 m/s2
    .min_accel = -3500,  // -3.5 m/s2
  };

  bool tx = true;

  // Check if msg is sent on BUS 0
  if (msg->bus == 0U) {
    // ACCEL: safety check on byte 1-2
    if (msg->addr == 0x343U) {
      int desired_accel = (msg->data[0] << 8) | msg->data[1];
      desired_accel = to_signed(desired_accel, 16);

      bool violation = false;
      if (toyota_secoc) {
        // SecOC cars move accel to 0x183. Only allow inactive accel on 0x343 to match stock behavior
        violation = desired_accel != TOYOTA_LONG_LIMITS.inactive_accel;
      }
      violation |= longitudinal_accel_checks(desired_accel, TOYOTA_LONG_LIMITS);

      // only ACC messages that cancel are allowed when openpilot is not controlling longitudinal
      if (toyota_stock_longitudinal) {
        bool cancel_req = GET_BIT(msg, 24U);
        if (!cancel_req) {
          violation = true;
        }
        if (desired_accel != TOYOTA_LONG_LIMITS.inactive_accel) {
          violation = true;
        }
      }

      if (violation) {
        tx = false;
      }
    }

    if (msg->addr == 0x183U) {
      int desired_accel = (msg->data[0] << 8) | msg->data[1];
      desired_accel = to_signed(desired_accel, 16);

      tx = !longitudinal_accel_checks(desired_accel, TOYOTA_LONG_LIMITS);
    }

    // AEB: block all actuation. only used when DSU is unplugged
    if (msg->addr == 0x283U) {
      // only allow the checksum, which is the last byte
      bool block = (GET_BYTES(msg, 0, 4) != 0U) || (msg->data[4] != 0U) || (msg->data[5] != 0U);
      if (block) {
        tx = false;
      }
    }

    // STEERING_LTA angle steering check
    if (msg->addr == 0x191U) {
      // check the STEER_REQUEST, STEER_REQUEST_2, TORQUE_WIND_DOWN, STEER_ANGLE_CMD signals
      bool lta_request = GET_BIT(msg, 0U);
      bool lta_request2 = GET_BIT(msg, 25U);
      int torque_wind_down = msg->data[5];
      int lta_angle = (msg->data[1] << 8) | msg->data[2];
      lta_angle = to_signed(lta_angle, 16);

      bool steer_control_enabled = lta_request || lta_request2;
      if (!toyota_lta) {
        // using torque (LKA), block LTA msgs with actuation requests
        if (steer_control_enabled || (lta_angle != 0) || (torque_wind_down != 0)) {
          tx = false;
        }
      } else {
        // check angle rate limits and inactive angle
        if (steer_angle_cmd_checks(lta_angle, steer_control_enabled, TOYOTA_ANGLE_STEERING_LIMITS)) {
          tx = false;
        }

        if (lta_request != lta_request2) {
          tx = false;
        }

        // TORQUE_WIND_DOWN is gated on steer request
        if (!steer_control_enabled && (torque_wind_down != 0)) {
          tx = false;
        }

        // TORQUE_WIND_DOWN can only be no or full torque
        if ((torque_wind_down != 0) && (torque_wind_down != 100)) {
          tx = false;
        }

        // check if we should wind down torque
        int driver_torque = SAFETY_MIN(SAFETY_ABS(torque_driver.min), SAFETY_ABS(torque_driver.max));
        if ((driver_torque > TOYOTA_LTA_MAX_DRIVER_TORQUE) && (torque_wind_down != 0)) {
          tx = false;
        }

        int eps_torque = SAFETY_MIN(SAFETY_ABS(torque_meas.min), SAFETY_ABS(torque_meas.max));
        if ((eps_torque > TOYOTA_LTA_MAX_MEAS_TORQUE) && (torque_wind_down != 0)) {
          tx = false;
        }
      }
    }

    // STEERING_LTA_2 angle steering check (SecOC)
    if (toyota_secoc && (msg->addr == 0x131U)) {
      // SecOC cars block any form of LTA actuation for now
      bool lta_request = GET_BIT(msg, 3U);  // STEERING_LTA_2.STEER_REQUEST
      bool lta_request2 = GET_BIT(msg, 0U);  // STEERING_LTA_2.STEER_REQUEST_2
      int lta_angle_msb = msg->data[2];  // STEERING_LTA_2.STEER_ANGLE_CMD (MSB)
      int lta_angle_lsb = msg->data[3];  // STEERING_LTA_2.STEER_ANGLE_CMD (LSB)

      bool actuation = lta_request || lta_request2 || (lta_angle_msb != 0) || (lta_angle_lsb != 0);
      if (actuation) {
        tx = false;
      }
    }

    // STEER: safety check on bytes 2-3
    if (msg->addr == 0x2E4U) {
      int desired_torque = (msg->data[1] << 8) | msg->data[2];
      desired_torque = to_signed(desired_torque, 16);
      bool steer_req = GET_BIT(msg, 0U);
      // When using LTA (angle control), assert no actuation on LKA message
      if (!toyota_lta) {
        if (steer_torque_cmd_checks(desired_torque, steer_req, TOYOTA_TORQUE_STEERING_LIMITS)) {
          tx = false;
        }
      } else {
        if ((desired_torque != 0) || steer_req) {
          tx = false;
        }
      }
    }
  }

  // UDS: Only tester present ("\x0F\x02\x3E\x00\x00\x00\x00\x00") allowed on diagnostics address
  if (msg->addr == 0x750U) {
    // this address is sub-addressed. only allow tester present to radar (0xF)
    bool invalid_uds_msg = (GET_BYTES(msg, 0, 4) != 0x003E020FU) || (GET_BYTES(msg, 4, 4) != 0x0U);
    if (invalid_uds_msg) {
      tx = 0;
    }
  }

  return tx;
}

static safety_config toyota_init(uint16_t param) {
  static const CanMsg TOYOTA_TX_MSGS[] = {
    TOYOTA_COMMON_TX_MSGS
  };

  static const CanMsg TOYOTA_SECOC_TX_MSGS[] = {
    TOYOTA_COMMON_SECOC_TX_MSGS
  };

  static const CanMsg TOYOTA_LONG_TX_MSGS[] = {
    TOYOTA_COMMON_LONG_TX_MSGS
  };

  static const CanMsg TOYOTA_SECOC_LONG_TX_MSGS[] = {
    TOYOTA_COMMON_SECOC_LONG_TX_MSGS
  };

  // safety param flags
  // first byte is for EPS factor, second is for flags
  const uint32_t TOYOTA_PARAM_OFFSET = 8U;
  const uint32_t TOYOTA_EPS_FACTOR = (1UL << TOYOTA_PARAM_OFFSET) - 1U;
  const uint32_t TOYOTA_PARAM_ALT_BRAKE = 1UL << TOYOTA_PARAM_OFFSET;
  const uint32_t TOYOTA_PARAM_STOCK_LONGITUDINAL = 2UL << TOYOTA_PARAM_OFFSET;
  const uint32_t TOYOTA_PARAM_LTA = 4UL << TOYOTA_PARAM_OFFSET;

#ifdef ALLOW_DEBUG
  const uint32_t TOYOTA_PARAM_SECOC = 8UL << TOYOTA_PARAM_OFFSET;
  toyota_secoc = GET_FLAG(param, TOYOTA_PARAM_SECOC);
#endif

  toyota_alt_brake = GET_FLAG(param, TOYOTA_PARAM_ALT_BRAKE);
  toyota_stock_longitudinal = GET_FLAG(param, TOYOTA_PARAM_STOCK_LONGITUDINAL);
  toyota_lta = GET_FLAG(param, TOYOTA_PARAM_LTA);
  toyota_dbc_eps_torque_factor = param & TOYOTA_EPS_FACTOR;

  safety_config ret;
  if (toyota_secoc) {
    if (toyota_stock_longitudinal) {
      SET_TX_MSGS(TOYOTA_SECOC_TX_MSGS, ret);
    } else {
      SET_TX_MSGS(TOYOTA_SECOC_LONG_TX_MSGS, ret);
    }
  } else {
    if (toyota_stock_longitudinal) {
      SET_TX_MSGS(TOYOTA_TX_MSGS, ret);
    } else {
      SET_TX_MSGS(TOYOTA_LONG_TX_MSGS, ret);
    }
  }

  if (toyota_secoc) {
    static RxCheck toyota_secoc_rx_checks[] = {
      TOYOTA_SECOC_RX_CHECKS
    };

    SET_RX_CHECKS(toyota_secoc_rx_checks, ret);
  } else if (toyota_lta) {
    // Check the quality flag for angle measurement when using LTA, since it's not set on TSS-P cars
    static RxCheck toyota_lta_rx_checks[] = {
      TOYOTA_RX_CHECKS(true)
    };

    SET_RX_CHECKS(toyota_lta_rx_checks, ret);
  } else {
    static RxCheck toyota_lka_rx_checks[] = {
      TOYOTA_RX_CHECKS(false)
    };
    static RxCheck toyota_lka_alt_brake_rx_checks[] = {
      TOYOTA_ALT_BRAKE_RX_CHECKS(false)
    };

    if (!toyota_alt_brake) {
      SET_RX_CHECKS(toyota_lka_rx_checks, ret);
    } else {
      SET_RX_CHECKS(toyota_lka_alt_brake_rx_checks, ret);
    }
  }

  return ret;
}

const safety_hooks toyota_hooks = {
  .init = toyota_init,
  .rx = toyota_rx_hook,
  .tx = toyota_tx_hook,
  .get_checksum = toyota_get_checksum,
  .compute_checksum = toyota_compute_checksum,
  .get_quality_flag_valid = toyota_get_quality_flag_valid,
};
