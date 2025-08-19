#pragma once

#include "opendbc/safety/safety_declarations.h"

#define SUBARU_STEERING_LIMITS_GENERATOR(steer_max, rate_up, rate_down)               \
  {                                                                                   \
    .max_torque = (steer_max),                                                        \
    .max_rt_delta = 940,                                                              \
    .max_rate_up = (rate_up),                                                         \
    .max_rate_down = (rate_down),                                                     \
    .driver_torque_multiplier = 50,                                                   \
    .driver_torque_allowance = 60,                                                    \
    .type = TorqueDriverLimited,                                                      \
    /* the EPS will temporary fault if the steering rate is too high, so we cut the   \
       the steering torque every 7 frames for 1 frame if the steering rate is high */ \
    .min_valid_request_frames = 7,                                                    \
    .max_invalid_request_frames = 1,                                                  \
    .min_valid_request_rt_interval = 144000,  /* 10% tolerance */                     \
    .has_steer_req_tolerance = true,                                                  \
  }

#define MSG_SUBARU_Brake_Status          0x13cU
#define MSG_SUBARU_CruiseControl         0x240U
#define MSG_SUBARU_Throttle              0x40U
#define MSG_SUBARU_Steering_Torque       0x119U
#define MSG_SUBARU_Wheel_Speeds          0x13aU

#define MSG_SUBARU_ES_LKAS               0x122U
#define MSG_SUBARU_ES_Brake              0x220U
#define MSG_SUBARU_ES_Distance           0x221U
#define MSG_SUBARU_ES_Status             0x222U
#define MSG_SUBARU_ES_DashStatus         0x321U
#define MSG_SUBARU_ES_LKAS_State         0x322U
#define MSG_SUBARU_ES_Infotainment       0x323U

#define MSG_SUBARU_ES_UDS_Request        0x787U

#define MSG_SUBARU_ES_HighBeamAssist     0x121U
#define MSG_SUBARU_ES_STATIC_1           0x22aU
#define MSG_SUBARU_ES_STATIC_2           0x325U

#define SUBARU_MAIN_BUS 0U
#define SUBARU_ALT_BUS  1U
#define SUBARU_CAM_BUS  2U

#define SUBARU_BASE_TX_MSGS(alt_bus, lkas_msg) \
  {lkas_msg,                     SUBARU_MAIN_BUS, 8, .check_relay = true},  \
  {MSG_SUBARU_ES_DashStatus,     SUBARU_MAIN_BUS, 8, .check_relay = true},  \
  {MSG_SUBARU_ES_LKAS_State,     SUBARU_MAIN_BUS, 8, .check_relay = true},  \
  {MSG_SUBARU_ES_Infotainment,   SUBARU_MAIN_BUS, 8, .check_relay = true},  \

#define SUBARU_COMMON_TX_MSGS(alt_bus) \
  {MSG_SUBARU_ES_Distance, alt_bus, 8, .check_relay = false}, \

#define SUBARU_COMMON_LONG_TX_MSGS(alt_bus) \
  {MSG_SUBARU_ES_Distance,       alt_bus,         8, .check_relay = true}, \
  {MSG_SUBARU_ES_Brake,          alt_bus,         8, .check_relay = true}, \
  {MSG_SUBARU_ES_Status,         alt_bus,         8, .check_relay = true}, \

#define SUBARU_GEN2_LONG_ADDITIONAL_TX_MSGS() \
  {MSG_SUBARU_ES_UDS_Request,    SUBARU_CAM_BUS,  8, .check_relay = false}, \
  {MSG_SUBARU_ES_HighBeamAssist, SUBARU_MAIN_BUS, 8, .check_relay = false}, \
  {MSG_SUBARU_ES_STATIC_1,       SUBARU_MAIN_BUS, 8, .check_relay = false}, \
  {MSG_SUBARU_ES_STATIC_2,       SUBARU_MAIN_BUS, 8, .check_relay = false}, \

#define SUBARU_COMMON_RX_CHECKS(alt_bus)                                                                                                         \
  {.msg = {{MSG_SUBARU_Throttle,        SUBARU_MAIN_BUS, 8, 100U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
  {.msg = {{MSG_SUBARU_Steering_Torque, SUBARU_MAIN_BUS, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \
  {.msg = {{MSG_SUBARU_Wheel_Speeds,    alt_bus,         8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \
  {.msg = {{MSG_SUBARU_Brake_Status,    alt_bus,         8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \
  {.msg = {{MSG_SUBARU_CruiseControl,   alt_bus,         8, 20U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  \

static bool subaru_gen2 = false;
static bool subaru_longitudinal = false;

static uint32_t subaru_get_checksum(const CANPacket_t *msg) {
  return (uint8_t)msg->data[0];
}

static uint8_t subaru_get_counter(const CANPacket_t *msg) {
  return (uint8_t)(msg->data[1] & 0xFU);
}

static uint32_t subaru_compute_checksum(const CANPacket_t *msg) {
  int len = GET_LEN(msg);
  uint8_t checksum = (uint8_t)(msg->addr) + (uint8_t)((unsigned int)(msg->addr) >> 8U);
  for (int i = 1; i < len; i++) {
    checksum += (uint8_t)msg->data[i];
  }
  return checksum;
}

static void subaru_rx_hook(const CANPacket_t *msg) {
  const unsigned int alt_main_bus = subaru_gen2 ? SUBARU_ALT_BUS : SUBARU_MAIN_BUS;

  if ((msg->addr == MSG_SUBARU_Steering_Torque) && (msg->bus == SUBARU_MAIN_BUS)) {
    int torque_driver_new;
    torque_driver_new = ((GET_BYTES(msg, 0, 4) >> 16) & 0x7FFU);
    torque_driver_new = -1 * to_signed(torque_driver_new, 11);
    update_sample(&torque_driver, torque_driver_new);

    int angle_meas_new = (GET_BYTES(msg, 4, 2) & 0xFFFFU);
    // convert Steering_Torque -> Steering_Angle to centidegrees, to match the ES_LKAS_ANGLE angle request units
    angle_meas_new = ROUND(to_signed(angle_meas_new, 16) * -2.17);
    update_sample(&angle_meas, angle_meas_new);
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  if ((msg->addr == MSG_SUBARU_CruiseControl) && (msg->bus == alt_main_bus)) {
    bool cruise_engaged = (msg->data[5] >> 1) & 1U;
    pcm_cruise_check(cruise_engaged);
  }

  // update vehicle moving with any non-zero wheel speed
  if ((msg->addr == MSG_SUBARU_Wheel_Speeds) && (msg->bus == alt_main_bus)) {
    uint32_t fr = (GET_BYTES(msg, 1, 3) >> 4) & 0x1FFFU;
    uint32_t rr = (GET_BYTES(msg, 3, 3) >> 1) & 0x1FFFU;
    uint32_t rl = (GET_BYTES(msg, 4, 3) >> 6) & 0x1FFFU;
    uint32_t fl = (GET_BYTES(msg, 6, 2) >> 3) & 0x1FFFU;

    vehicle_moving = (fr > 0U) || (rr > 0U) || (rl > 0U) || (fl > 0U);

    UPDATE_VEHICLE_SPEED((fr + rr + rl + fl) / 4.0 * 0.057 * KPH_TO_MS);
  }

  if ((msg->addr == MSG_SUBARU_Brake_Status) && (msg->bus == alt_main_bus)) {
    brake_pressed = (msg->data[7] >> 6) & 1U;
  }

  if ((msg->addr == MSG_SUBARU_Throttle) && (msg->bus == SUBARU_MAIN_BUS)) {
    gas_pressed = msg->data[4] != 0U;
  }
}

static bool subaru_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits SUBARU_STEERING_LIMITS      = SUBARU_STEERING_LIMITS_GENERATOR(2047, 50, 70);
  const TorqueSteeringLimits SUBARU_GEN2_STEERING_LIMITS = SUBARU_STEERING_LIMITS_GENERATOR(1000, 40, 40);

  const LongitudinalLimits SUBARU_LONG_LIMITS = {
    .min_gas = 808,       // appears to be engine braking
    .max_gas = 3400,      // approx  2 m/s^2 when maxing cruise_rpm and cruise_throttle
    .inactive_gas = 1818, // this is zero acceleration
    .max_brake = 600,     // approx -3.5 m/s^2

    .min_transmission_rpm = 0,
    .max_transmission_rpm = 3600,
  };

  bool tx = true;
  bool violation = false;

  // steer cmd checks
  if (msg->addr == MSG_SUBARU_ES_LKAS) {
    int desired_torque = ((GET_BYTES(msg, 0, 4) >> 16) & 0x1FFFU);
    desired_torque = -1 * to_signed(desired_torque, 13);

    bool steer_req = (msg->data[3] >> 5) & 1U;

    const TorqueSteeringLimits limits = subaru_gen2 ? SUBARU_GEN2_STEERING_LIMITS : SUBARU_STEERING_LIMITS;
    violation |= steer_torque_cmd_checks(desired_torque, steer_req, limits);
  }

  // check es_brake brake_pressure limits
  if (msg->addr == MSG_SUBARU_ES_Brake) {
    int es_brake_pressure = GET_BYTES(msg, 2, 2);
    violation |= longitudinal_brake_checks(es_brake_pressure, SUBARU_LONG_LIMITS);
  }

  // check es_distance cruise_throttle limits
  if (msg->addr == MSG_SUBARU_ES_Distance) {
    int cruise_throttle = (GET_BYTES(msg, 2, 2) & 0x1FFFU);
    bool cruise_cancel = (msg->data[7] >> 0) & 1U;

    if (subaru_longitudinal) {
      violation |= longitudinal_gas_checks(cruise_throttle, SUBARU_LONG_LIMITS);
    } else {
      // If openpilot is not controlling long, only allow ES_Distance for cruise cancel requests,
      // (when Cruise_Cancel is true, and Cruise_Throttle is inactive)
      violation |= (cruise_throttle != SUBARU_LONG_LIMITS.inactive_gas);
      violation |= (!cruise_cancel);
    }
  }

  // check es_status transmission_rpm limits
  if (msg->addr == MSG_SUBARU_ES_Status) {
    int transmission_rpm = (GET_BYTES(msg, 2, 2) & 0x1FFFU);
    violation |= longitudinal_transmission_rpm_checks(transmission_rpm, SUBARU_LONG_LIMITS);
  }

  if (msg->addr == MSG_SUBARU_ES_UDS_Request) {
    // tester present ('\x02\x3E\x80\x00\x00\x00\x00\x00') is allowed for gen2 longitudinal to keep eyesight disabled
    bool is_tester_present = (GET_BYTES(msg, 0, 4) == 0x00803E02U) && (GET_BYTES(msg, 4, 4) == 0x0U);

    // reading ES button data by identifier (b'\x03\x22\x11\x30\x00\x00\x00\x00') is also allowed (DID 0x1130)
    bool is_button_rdbi = (GET_BYTES(msg, 0, 4) == 0x30112203U) && (GET_BYTES(msg, 4, 4) == 0x0U);

    violation |= !(is_tester_present || is_button_rdbi);
  }

  if (violation){
    tx = false;
  }
  return tx;
}

static safety_config subaru_init(uint16_t param) {
  static const CanMsg SUBARU_TX_MSGS[] = {
    SUBARU_BASE_TX_MSGS(SUBARU_MAIN_BUS, MSG_SUBARU_ES_LKAS)
    SUBARU_COMMON_TX_MSGS(SUBARU_MAIN_BUS)
  };

  static const CanMsg SUBARU_LONG_TX_MSGS[] = {
    SUBARU_BASE_TX_MSGS(SUBARU_MAIN_BUS, MSG_SUBARU_ES_LKAS)
    SUBARU_COMMON_LONG_TX_MSGS(SUBARU_MAIN_BUS)
  };

  static const CanMsg SUBARU_GEN2_TX_MSGS[] = {
    SUBARU_BASE_TX_MSGS(SUBARU_ALT_BUS, MSG_SUBARU_ES_LKAS)
    SUBARU_COMMON_TX_MSGS(SUBARU_ALT_BUS)
  };

  static const CanMsg SUBARU_GEN2_LONG_TX_MSGS[] = {
    SUBARU_BASE_TX_MSGS(SUBARU_ALT_BUS, MSG_SUBARU_ES_LKAS)
    SUBARU_COMMON_LONG_TX_MSGS(SUBARU_ALT_BUS)
    SUBARU_GEN2_LONG_ADDITIONAL_TX_MSGS()
  };

  static RxCheck subaru_rx_checks[] = {
    SUBARU_COMMON_RX_CHECKS(SUBARU_MAIN_BUS)
  };

  static RxCheck subaru_gen2_rx_checks[] = {
    SUBARU_COMMON_RX_CHECKS(SUBARU_ALT_BUS)
  };

  const uint16_t SUBARU_PARAM_GEN2 = 1;

  subaru_gen2 = GET_FLAG(param, SUBARU_PARAM_GEN2);

#ifdef ALLOW_DEBUG
  const uint16_t SUBARU_PARAM_LONGITUDINAL = 2;
  subaru_longitudinal = GET_FLAG(param, SUBARU_PARAM_LONGITUDINAL);
#endif

  safety_config ret;
  if (subaru_gen2) {
    ret = subaru_longitudinal ? BUILD_SAFETY_CFG(subaru_gen2_rx_checks, SUBARU_GEN2_LONG_TX_MSGS) : \
                                BUILD_SAFETY_CFG(subaru_gen2_rx_checks, SUBARU_GEN2_TX_MSGS);
  } else {
    ret = subaru_longitudinal ? BUILD_SAFETY_CFG(subaru_rx_checks, SUBARU_LONG_TX_MSGS) : \
                                BUILD_SAFETY_CFG(subaru_rx_checks, SUBARU_TX_MSGS);
  }
  return ret;
}

const safety_hooks subaru_hooks = {
  .init = subaru_init,
  .rx = subaru_rx_hook,
  .tx = subaru_tx_hook,
  .get_counter = subaru_get_counter,
  .get_checksum = subaru_get_checksum,
  .compute_checksum = subaru_compute_checksum,
};
