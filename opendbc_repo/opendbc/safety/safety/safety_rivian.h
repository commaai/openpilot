#pragma once

#include "safety_declarations.h"

static bool rivian_longitudinal = false;

static void rivian_rx_hook(const CANPacket_t *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  if (bus == 0)  {
    // Vehicle speed
    if (addr == 0x208) {
      vehicle_moving = GET_BYTE(to_push, 6) | GET_BYTE(to_push, 7);
    }

    // Driver torque
    if (addr == 0x380) {
      int torque_driver_new = (((GET_BYTE(to_push, 2) << 4) | (GET_BYTE(to_push, 3) >> 4))) - 2050U;
      update_sample(&torque_driver, torque_driver_new);
    }

    // Gas pressed
    if (addr == 0x150) {
      gas_pressed = GET_BYTE(to_push, 3) | (GET_BYTE(to_push, 4) & 0xC0U);
    }

    // Brake pressed
    if (addr == 0x38f) {
      brake_pressed = GET_BIT(to_push, 23U);
    }

    generic_rx_checks(addr == 0x120);  // ACM_lkaHbaCmd
    if (rivian_longitudinal) {
      generic_rx_checks(addr == 0x160);  // ACM_longitudinalRequest
    }
  }

  if (bus == 2) {
    // Cruise state
    if (addr == 0x100) {
      pcm_cruise_check(GET_BIT(to_push, 21U));
    }
  }
}

static bool rivian_tx_hook(const CANPacket_t *to_send) {
  const TorqueSteeringLimits RIVIAN_STEERING_LIMITS = {
    .max_steer = 250,
    .max_rate_up = 3,
    .max_rate_down = 5,
    .max_rt_delta = 125,
    .max_rt_interval = 250000,
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
      if (rivian_longitudinal) {
        int raw_accel = ((GET_BYTE(to_send, 2) << 3) | (GET_BYTE(to_send, 3) >> 5)) - 1024U;
        if (longitudinal_accel_checks(raw_accel, RIVIAN_LONG_LIMITS)) {
          tx = false;
        }
      } else {
        tx = false;
      }
    }
  }

  return tx;
}

static int rivian_fwd_hook(int bus, int addr) {
  int bus_fwd = -1;
  bool block_msg = false;

  if (bus == 0) {
    // SCCM_WheelTouch
    if (addr == 0x321) {
      block_msg = true;
    }

    if (!block_msg) {
      bus_fwd = 2;
    }
  }

  if (bus == 2) {
    // ACM_lkaHbaCmd
    if (addr == 0x120) {
      block_msg = true;
    }

    // ACM_longitudinalRequest
    if (rivian_longitudinal && (addr == 0x160)) {
      block_msg = true;
    }

    if (!block_msg) {
      bus_fwd = 0;
    }
  }

  return bus_fwd;
}

static safety_config rivian_init(uint16_t param) {
  // 0x120 = ACM_lkaHbaCmd, 0x160 = ACM_longitudinalRequest, 0x321 = SCCM_WheelTouch
  static const CanMsg RIVIAN_TX_MSGS[] = {{0x120, 0, 8}, {0x321, 2, 7}};
  static const CanMsg RIVIAN_LONG_TX_MSGS[] = {{0x120, 0, 8}, {0x321, 2, 7}, {0x160, 0, 5}};

  static RxCheck rivian_rx_checks[] = {
    {.msg = {{0x208, 0, 8, .frequency = 50U}, { 0 }, { 0 }}},   // ESP_Status (speed)
    {.msg = {{0x150, 0, 7, .frequency = 50U}, { 0 }, { 0 }}},   // VDM_PropStatus (gas pedal)
    {.msg = {{0x38f, 0, 6, .frequency = 50U}, { 0 }, { 0 }}},   // iBESP2 (brakes)
    {.msg = {{0x100, 2, 8, .frequency = 100U}, { 0 }, { 0 }}},  // ACM_Status (cruise state)
  };

  UNUSED(param);
  #ifdef ALLOW_DEBUG
    const int FLAG_RIVIAN_LONG_CONTROL = 1;
    rivian_longitudinal = GET_FLAG(param, FLAG_RIVIAN_LONG_CONTROL);
  #endif

  return rivian_longitudinal ? BUILD_SAFETY_CFG(rivian_rx_checks, RIVIAN_LONG_TX_MSGS) : \
                               BUILD_SAFETY_CFG(rivian_rx_checks, RIVIAN_TX_MSGS);
}

const safety_hooks rivian_hooks = {
  .init = rivian_init,
  .rx = rivian_rx_hook,
  .tx = rivian_tx_hook,
  .fwd = rivian_fwd_hook,
};
