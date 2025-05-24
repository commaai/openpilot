#pragma once

#include "opendbc/safety/safety_declarations.h"

#define RIVIAN_MAX_SPEED_DELTA 2.0  // m/s

static uint8_t rivian_get_counter(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t cnt = 0;
  if ((addr == 0x208) || (addr == 0x150)) {
    // Signal: ESP_Status_Counter, VDM_PropStatus_Counter
    cnt = GET_BYTE(to_push, 1) & 0xFU;
  } else {
  }
  return cnt;
}

static uint32_t rivian_get_checksum(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t chksum = 0;
  if ((addr == 0x208) || (addr == 0x150)) {
    // Signal: ESP_Status_Checksum, VDM_PropStatus_Checksum
    chksum = GET_BYTE(to_push, 0);
  } else {
  }
  return chksum;
}

static uint8_t _rivian_compute_checksum(const CANPacket_t *to_push, uint8_t poly, uint8_t xor_output) {
  int len = GET_LEN(to_push);

  uint8_t crc = 0;
  // Skip the checksum byte
  for (int i = 1; i < len; i++) {
    crc ^= GET_BYTE(to_push, i);
    for (int j = 0; j < 8; j++) {
      if ((crc & 0x80U) != 0U) {
        crc = (crc << 1) ^ poly;
      } else {
        crc <<= 1;
      }
    }
  }
  return crc ^ xor_output;
}

static uint32_t rivian_compute_checksum(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t chksum = 0;
  if (addr == 0x208) {
    chksum = _rivian_compute_checksum(to_push, 0x1D, 0xB1);
  } else if (addr == 0x150) {
    chksum = _rivian_compute_checksum(to_push, 0x1D, 0x9A);
  } else {
  }
  return chksum;
}

static bool rivian_get_quality_flag_valid(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  bool valid = false;
  if (addr == 0x208) {
    valid = ((GET_BYTE(to_push, 3) >> 3) & 0x3U) == 0x1U;  // ESP_Vehicle_Speed_Q
  } else if (addr == 0x150) {
    valid = (GET_BYTE(to_push, 1) >> 6) == 0x1U;  // VDM_VehicleSpeedQ
  } else {
  }
  return valid;
}

static void rivian_rx_hook(const CANPacket_t *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  if (bus == 0)  {
    // Vehicle speed
    if (addr == 0x208) {
      float speed = ((GET_BYTE(to_push, 6) << 8) | GET_BYTE(to_push, 7)) * 0.01;
      vehicle_moving = speed > 0.0;
      UPDATE_VEHICLE_SPEED(speed / 3.6);
    }

    // Gas pressed and second speed source for variable torque limit
    if (addr == 0x150) {
      gas_pressed = GET_BYTE(to_push, 3) | (GET_BYTE(to_push, 4) & 0xC0U);

      // Disable controls if speeds from VDM and ESP ECUs are too far apart.
      float vdm_speed = ((GET_BYTE(to_push, 5) << 8) | GET_BYTE(to_push, 6)) * 0.01 / 3.6;
      bool is_invalid_speed = ABS(vdm_speed - ((float)vehicle_speed.values[0] / VEHICLE_SPEED_FACTOR)) > RIVIAN_MAX_SPEED_DELTA;
      // TODO: this should generically cause rx valid to fall until re-enable
      if (is_invalid_speed) {
        controls_allowed = false;
      }
    }

    // Driver torque
    if (addr == 0x380) {
      int torque_driver_new = (((GET_BYTE(to_push, 2) << 4) | (GET_BYTE(to_push, 3) >> 4))) - 2050U;
      update_sample(&torque_driver, torque_driver_new);
    }

    // Brake pressed
    if (addr == 0x38f) {
      brake_pressed = GET_BIT(to_push, 23U);
    }
  }

  if (bus == 2) {
    // Cruise state
    if (addr == 0x100) {
      const int feature_status = GET_BYTE(to_push, 2) >> 5U;
      pcm_cruise_check(feature_status == 1);
    }
  }
}

static bool rivian_tx_hook(const CANPacket_t *to_send) {
  // Rivian utilizes more torque at low speed to maintain the same lateral accel
  const TorqueSteeringLimits RIVIAN_STEERING_LIMITS = {
    .max_torque = 350,
    .dynamic_max_torque = true,
    .max_torque_lookup = {
      {9., 17., 17.},
      {350, 250, 250},
    },
    .max_rate_up = 3,
    .max_rate_down = 5,
    .max_rt_delta = 125,
    .driver_torque_multiplier = 2,
    .driver_torque_allowance = 100,
    .type = TorqueDriverLimited,
  };

  const LongitudinalLimits RIVIAN_LONG_LIMITS = {
    .max_accel = 200,
    .min_accel = -350,
    .inactive_accel = 0,
  };

  bool tx = true;
  int bus = GET_BUS(to_send);

  if (bus == 0) {
    int addr = GET_ADDR(to_send);

    // Steering control
    if (addr == 0x120) {
      int desired_torque = ((GET_BYTE(to_send, 2) << 3U) | (GET_BYTE(to_send, 3) >> 5U)) - 1024U;
      bool steer_req = GET_BIT(to_send, 28U);

      if (steer_torque_cmd_checks(desired_torque, steer_req, RIVIAN_STEERING_LIMITS)) {
        tx = false;
      }
    }

    // Longitudinal control
    if (addr == 0x160) {
      int raw_accel = ((GET_BYTE(to_send, 2) << 3) | (GET_BYTE(to_send, 3) >> 5)) - 1024U;
      if (longitudinal_accel_checks(raw_accel, RIVIAN_LONG_LIMITS)) {
        tx = false;
      }
    }
  }

  return tx;
}

static safety_config rivian_init(uint16_t param) {
  // SCCM_WheelTouch: for hiding hold wheel alert
  // VDM_AdasSts: for canceling stock ACC
  // 0x120 = ACM_lkaHbaCmd, 0x321 = SCCM_WheelTouch, 0x162 = VDM_AdasSts
  static const CanMsg RIVIAN_TX_MSGS[] = {{0x120, 0, 8, .check_relay = true}, {0x321, 2, 7, .check_relay = true}, {0x162, 2, 8, .check_relay = true}};
  // 0x160 = ACM_longitudinalRequest
  static const CanMsg RIVIAN_LONG_TX_MSGS[] = {{0x120, 0, 8, .check_relay = true}, {0x321, 2, 7, .check_relay = true}, {0x160, 0, 5, .check_relay = true}};

  static RxCheck rivian_rx_checks[] = {
    {.msg = {{0x208, 0, 8, .frequency = 50U, .max_counter = 14U, .quality_flag = true}, { 0 }, { 0 }}},          // ESP_Status (speed)
    {.msg = {{0x150, 0, 7, .frequency = 50U, .max_counter = 14U, .quality_flag = true}, { 0 }, { 0 }}},          // VDM_PropStatus (gas pedal & 2nd speed)
    {.msg = {{0x380, 0, 5, .frequency = 100U, .ignore_checksum = true, .ignore_counter = true}, { 0 }, { 0 }}},  // EPAS_SystemStatus (driver torque)
    {.msg = {{0x38f, 0, 6, .frequency = 50U, .ignore_checksum = true, .ignore_counter = true}, { 0 }, { 0 }}},   // iBESP2 (brakes)
    {.msg = {{0x100, 2, 8, .frequency = 100U, .ignore_checksum = true, .ignore_counter = true}, { 0 }, { 0 }}},  // ACM_Status (cruise state)
  };

  bool rivian_longitudinal = false;

  UNUSED(param);
  #ifdef ALLOW_DEBUG
    const int FLAG_RIVIAN_LONG_CONTROL = 1;
    rivian_longitudinal = GET_FLAG(param, FLAG_RIVIAN_LONG_CONTROL);
  #endif

  // FIXME: cppcheck thinks that rivian_longitudinal is always false. This is not true
  // if ALLOW_DEBUG is defined but cppcheck is run without ALLOW_DEBUG
  // cppcheck-suppress knownConditionTrueFalse
  return rivian_longitudinal ? BUILD_SAFETY_CFG(rivian_rx_checks, RIVIAN_LONG_TX_MSGS) : \
                               BUILD_SAFETY_CFG(rivian_rx_checks, RIVIAN_TX_MSGS);
}

const safety_hooks rivian_hooks = {
  .init = rivian_init,
  .rx = rivian_rx_hook,
  .tx = rivian_tx_hook,
  .get_counter = rivian_get_counter,
  .get_checksum = rivian_get_checksum,
  .compute_checksum = rivian_compute_checksum,
  .get_quality_flag_valid = rivian_get_quality_flag_valid,
};
