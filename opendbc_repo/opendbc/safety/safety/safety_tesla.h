#pragma once

#include "safety_declarations.h"

static bool tesla_longitudinal = false;
static bool tesla_stock_aeb = false;

static void tesla_rx_hook(const CANPacket_t *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  if (bus == 0) {
    // Steering angle: (0.1 * val) - 819.2 in deg.
    if (addr == 0x370) {
      // Store it 1/10 deg to match steering request
      int angle_meas_new = (((GET_BYTE(to_push, 4) & 0x3FU) << 8) | GET_BYTE(to_push, 5)) - 8192U;
      update_sample(&angle_meas, angle_meas_new);
    }

    // Vehicle speed
    if (addr == 0x257) {
      // Vehicle speed: ((val * 0.08) - 40) / MS_TO_KPH
      float speed = ((((GET_BYTE(to_push, 2) << 4) | (GET_BYTE(to_push, 1) >> 4)) * 0.08) - 40) / 3.6;
      UPDATE_VEHICLE_SPEED(speed);
    }

    // Gas pressed
    if (addr == 0x118) {
      gas_pressed = (GET_BYTE(to_push, 4) != 0U);
    }

    // Brake pressed
    if (addr == 0x39d) {
      brake_pressed = (GET_BYTE(to_push, 2) & 0x03U) == 2U;
    }

    // Cruise state
    if (addr == 0x286) {
      int cruise_state = (GET_BYTE(to_push, 1) >> 4) & 0x07U;
      bool cruise_engaged = (cruise_state == 2) ||  // ENABLED
                            (cruise_state == 3) ||  // STANDSTILL
                            (cruise_state == 4) ||  // OVERRIDE
                            (cruise_state == 6) ||  // PRE_FAULT
                            (cruise_state == 7);    // PRE_CANCEL

      vehicle_moving = cruise_state != 3; // STANDSTILL
      pcm_cruise_check(cruise_engaged);
    }
  }

  if (bus == 2) {
    if (tesla_longitudinal && (addr == 0x2b9)) {
      // "AEB_ACTIVE"
      tesla_stock_aeb = (GET_BYTE(to_push, 2) & 0x03U) == 1U;
    }
  }

  generic_rx_checks((addr == 0x488) && (bus == 0));  // DAS_steeringControl
  generic_rx_checks((addr == 0x27d) && (bus == 0));  // APS_eacMonitor

  if (tesla_longitudinal) {
    generic_rx_checks((addr == 0x2b9) && (bus == 0));
  }
}


static bool tesla_tx_hook(const CANPacket_t *to_send) {
  const AngleSteeringLimits TESLA_STEERING_LIMITS = {
    .max_angle = 3600,  // 360 deg, EPAS faults above this
    .angle_deg_to_can = 10,
    .angle_rate_up_lookup = {
      {0., 5., 25.},
      {2.5, 1.5, 0.2}
    },
    .angle_rate_down_lookup = {
      {0., 5., 25.},
      {5., 2.0, 0.3}
    },
  };

  const LongitudinalLimits TESLA_LONG_LIMITS = {
    .max_accel = 425,       // 2 m/s^2
    .min_accel = 288,       // -3.48 m/s^2
    .inactive_accel = 375,  // 0. m/s^2
  };

  bool tx = true;
  int addr = GET_ADDR(to_send);
  bool violation = false;

  // Steering control: (0.1 * val) - 1638.35 in deg.
  if (addr == 0x488) {
    // We use 1/10 deg as a unit here
    int raw_angle_can = ((GET_BYTE(to_send, 0) & 0x7FU) << 8) | GET_BYTE(to_send, 1);
    int desired_angle = raw_angle_can - 16384;
    int steer_control_type = GET_BYTE(to_send, 2) >> 6;
    bool steer_control_enabled = (steer_control_type != 0) &&  // NONE
                                 (steer_control_type != 3);    // DISABLED

    if (steer_angle_cmd_checks(desired_angle, steer_control_enabled, TESLA_STEERING_LIMITS)) {
      violation = true;
    }
  }

  // DAS_control: longitudinal control message
  if (addr == 0x2b9) {
    // No AEB events may be sent by openpilot
    int aeb_event = GET_BYTE(to_send, 2) & 0x03U;
    if (aeb_event != 0) {
      violation = true;
    }

    int raw_accel_max = ((GET_BYTE(to_send, 6) & 0x1FU) << 4) | (GET_BYTE(to_send, 5) >> 4);
    int raw_accel_min = ((GET_BYTE(to_send, 5) & 0x0FU) << 5) | (GET_BYTE(to_send, 4) >> 3);
    int acc_state = GET_BYTE(to_send, 1) >> 4;

    if (tesla_longitudinal) {
      // Don't send messages when the stock AEB system is active
      if (tesla_stock_aeb) {
        violation = true;
      }

      // Prevent both acceleration from being negative, as this could cause the car to reverse after coming to standstill
      if ((raw_accel_max < TESLA_LONG_LIMITS.inactive_accel) && (raw_accel_min < TESLA_LONG_LIMITS.inactive_accel)) {
        violation = true;
      }

      // Don't allow any acceleration limits above the safety limits
      violation |= longitudinal_accel_checks(raw_accel_max, TESLA_LONG_LIMITS);
      violation |= longitudinal_accel_checks(raw_accel_min, TESLA_LONG_LIMITS);
    } else {
      // does allowing cancel here disrupt stock AEB? TODO: find out and add safety or remove comment
      // Can only send cancel longitudinal messages when not controlling longitudinal
      if (acc_state != 13) {  // ACC_CANCEL_GENERIC_SILENT
        violation = true;
      }

      // No actuation is allowed when not controlling longitudinal
      if ((raw_accel_max != TESLA_LONG_LIMITS.inactive_accel) || (raw_accel_min != TESLA_LONG_LIMITS.inactive_accel)) {
        violation = true;
      }
    }
  }

  if (violation) {
    tx = false;
  }

  return tx;
}

static int tesla_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;

  if (bus_num == 0) {
    // Party to autopilot
    bus_fwd = 2;
  }

  if (bus_num == 2) {
    bool block_msg = false;
    // DAS_steeringControl, APS_eacMonitor
    if ((addr == 0x488) || (addr == 0x27d)) {
      block_msg = true;
    }

    // DAS_control
    if (tesla_longitudinal && (addr == 0x2b9) && !tesla_stock_aeb) {
      block_msg = true;
    }

    if (!block_msg) {
      bus_fwd = 0;
    }
  }

  return bus_fwd;
}

static safety_config tesla_init(uint16_t param) {

  static const CanMsg TESLA_M3_Y_TX_MSGS[] = {
    {0x488, 0, 4},  // DAS_steeringControl
    {0x2b9, 0, 8},  // DAS_control
    {0x27D, 0, 3},  // APS_eacMonitor
  };

  UNUSED(param);
#ifdef ALLOW_DEBUG
  const int TESLA_FLAG_LONGITUDINAL_CONTROL = 1;
  tesla_longitudinal = GET_FLAG(param, TESLA_FLAG_LONGITUDINAL_CONTROL);
#endif

  tesla_stock_aeb = false;

  static RxCheck tesla_model3_y_rx_checks[] = {
    {.msg = {{0x2b9, 2, 8, .frequency = 25U}, { 0 }, { 0 }}},   // DAS_control
    {.msg = {{0x257, 0, 8, .frequency = 50U}, { 0 }, { 0 }}},   // DI_speed (speed in kph)
    {.msg = {{0x370, 0, 8, .frequency = 100U}, { 0 }, { 0 }}},  // EPAS3S_internalSAS (steering angle)
    {.msg = {{0x118, 0, 8, .frequency = 100U}, { 0 }, { 0 }}},  // DI_systemStatus (gas pedal)
    {.msg = {{0x39d, 0, 5, .frequency = 25U}, { 0 }, { 0 }}},   // IBST_status (brakes)
    {.msg = {{0x286, 0, 8, .frequency = 10U}, { 0 }, { 0 }}},   // DI_state (acc state)
    {.msg = {{0x311, 0, 7, .frequency = 10U}, { 0 }, { 0 }}},   // UI_warning (blinkers, buckle switch & doors)
  };

  return BUILD_SAFETY_CFG(tesla_model3_y_rx_checks, TESLA_M3_Y_TX_MSGS);
}

const safety_hooks tesla_hooks = {
  .init = tesla_init,
  .rx = tesla_rx_hook,
  .tx = tesla_tx_hook,
  .fwd = tesla_fwd_hook,
};
