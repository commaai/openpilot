#pragma once

#include "opendbc/safety/safety_declarations.h"

// TODO: do checksum and counter checks. Add correct timestep, 0.1s for now.
#define GM_COMMON_RX_CHECKS \
    {.msg = {{0x184, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
    {.msg = {{0x34A, 0, 5, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
    {.msg = {{0x1E1, 0, 7, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
    {.msg = {{0xBE, 0, 6, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},    /* Volt, Silverado, Acadia Denali */ \
             {0xBE, 0, 7, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},    /* Bolt EUV */ \
             {0xBE, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}}},  /* Escalade */ \
    {.msg = {{0x1C4, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
    {.msg = {{0xC9, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \

static const LongitudinalLimits *gm_long_limits;

enum {
  GM_BTN_UNPRESS = 1,
  GM_BTN_RESUME = 2,
  GM_BTN_SET = 3,
  GM_BTN_CANCEL = 6,
};

typedef enum {
  GM_ASCM,
  GM_CAM
} GmHardware;
static GmHardware gm_hw = GM_ASCM;
static bool gm_pcm_cruise = false;

static void gm_rx_hook(const CANPacket_t *msg) {
  const int GM_STANDSTILL_THRSLD = 10;  // 0.311kph

  if (msg->bus == 0U) {
    if (msg->addr == 0x184U) {
      int torque_driver_new = ((msg->data[6] & 0x7U) << 8) | msg->data[7];
      torque_driver_new = to_signed(torque_driver_new, 11);
      // update array of samples
      update_sample(&torque_driver, torque_driver_new);
    }

    // sample rear wheel speeds
    if (msg->addr == 0x34AU) {
      int left_rear_speed = (msg->data[0] << 8) | msg->data[1];
      int right_rear_speed = (msg->data[2] << 8) | msg->data[3];
      vehicle_moving = (left_rear_speed > GM_STANDSTILL_THRSLD) || (right_rear_speed > GM_STANDSTILL_THRSLD);
    }

    // ACC steering wheel buttons (GM_CAM is tied to the PCM)
    if ((msg->addr == 0x1E1U) && !gm_pcm_cruise) {
      int button = (msg->data[5] & 0x70U) >> 4;

      // enter controls on falling edge of set or rising edge of resume (avoids fault)
      bool set = (button != GM_BTN_SET) && (cruise_button_prev == GM_BTN_SET);
      bool res = (button == GM_BTN_RESUME) && (cruise_button_prev != GM_BTN_RESUME);
      if (set || res) {
        controls_allowed = true;
      }

      // exit controls on cancel press
      if (button == GM_BTN_CANCEL) {
        controls_allowed = false;
      }

      cruise_button_prev = button;
    }

    // Reference for brake pressed signals:
    // https://github.com/commaai/openpilot/blob/master/selfdrive/car/gm/carstate.py
    if ((msg->addr == 0xBEU) && (gm_hw == GM_ASCM)) {
      brake_pressed = msg->data[1] >= 8U;
    }

    if ((msg->addr == 0xC9U) && (gm_hw == GM_CAM)) {
      brake_pressed = GET_BIT(msg, 40U);
    }

    if (msg->addr == 0x1C4U) {
      gas_pressed = msg->data[5] != 0U;

      // enter controls on rising edge of ACC, exit controls when ACC off
      if (gm_pcm_cruise) {
        bool cruise_engaged = (msg->data[1] >> 5) != 0U;
        pcm_cruise_check(cruise_engaged);
      }
    }

    if (msg->addr == 0xBDU) {
      regen_braking = (msg->data[0] >> 4) != 0U;
    }
  }
}

static bool gm_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits GM_STEERING_LIMITS = {
    .max_torque = 300,
    .max_rate_up = 10,
    .max_rate_down = 15,
    .driver_torque_allowance = 65,
    .driver_torque_multiplier = 4,
    .max_rt_delta = 128,
    .type = TorqueDriverLimited,
  };

  bool tx = true;

  // BRAKE: safety check
  if (msg->addr == 0x315U) {
    int brake = ((msg->data[0] & 0xFU) << 8) + msg->data[1];
    brake = (0x1000 - brake) & 0xFFF;
    if (longitudinal_brake_checks(brake, *gm_long_limits)) {
      tx = false;
    }
  }

  // LKA STEER: safety check
  if (msg->addr == 0x180U) {
    int desired_torque = ((msg->data[0] & 0x7U) << 8) + msg->data[1];
    desired_torque = to_signed(desired_torque, 11);

    bool steer_req = GET_BIT(msg, 3U);

    if (steer_torque_cmd_checks(desired_torque, steer_req, GM_STEERING_LIMITS)) {
      tx = false;
    }
  }

  // GAS/REGEN: safety check
  if (msg->addr == 0x2CBU) {
    bool apply = GET_BIT(msg, 0U);
    // convert float CAN signal to an int for gas checks: 22534 / 0.125 = 180272
    int gas_regen = (((msg->data[1] & 0x7U) << 16) | (msg->data[2] << 8) | msg->data[3]) - 180272U;

    bool violation = false;
    // Allow apply bit in pre-enabled and overriding states
    violation |= !controls_allowed && apply;
    violation |= longitudinal_gas_checks(gas_regen, *gm_long_limits);

    if (violation) {
      tx = false;
    }
  }

  // BUTTONS: used for resume spamming and cruise cancellation with stock longitudinal
  if ((msg->addr == 0x1E1U) && gm_pcm_cruise) {
    int button = (msg->data[5] >> 4) & 0x7U;

    bool allowed_cancel = (button == 6) && cruise_engaged_prev;
    if (!allowed_cancel) {
      tx = false;
    }
  }

  return tx;
}

static safety_config gm_init(uint16_t param) {
  const uint16_t GM_PARAM_HW_CAM = 1;
  const uint16_t GM_PARAM_EV = 4;

  // common safety checks assume unscaled integer values
  static const int GM_GAS_TO_CAN = 8;  // 1 / 0.125

  static const LongitudinalLimits GM_ASCM_LONG_LIMITS = {
    .max_gas = 1018 * GM_GAS_TO_CAN,
    .min_gas = -650 * GM_GAS_TO_CAN,
    .inactive_gas = -650 * GM_GAS_TO_CAN,
    .max_brake = 400,
  };

  static const CanMsg GM_ASCM_TX_MSGS[] = {{0x180, 0, 4, .check_relay = true}, {0x409, 0, 7, .check_relay = false}, {0x40A, 0, 7, .check_relay = false}, {0x2CB, 0, 8, .check_relay = true}, {0x370, 0, 6, .check_relay = false},  // pt bus
                                           {0xA1, 1, 7, .check_relay = false}, {0x306, 1, 8, .check_relay = false}, {0x308, 1, 7, .check_relay = false}, {0x310, 1, 2, .check_relay = false},   // obs bus
                                           {0x315, 2, 5, .check_relay = false}};  // ch bus


  static const LongitudinalLimits GM_CAM_LONG_LIMITS = {
    .max_gas = 1346 * GM_GAS_TO_CAN,
    .min_gas = -540 * GM_GAS_TO_CAN,
    .inactive_gas = -500 * GM_GAS_TO_CAN,
    .max_brake = 400,
  };

  // block PSCMStatus (0x184); forwarded through openpilot to hide an alert from the camera
  static const CanMsg GM_CAM_LONG_TX_MSGS[] = {{0x180, 0, 4, .check_relay = true}, {0x315, 0, 5, .check_relay = true}, {0x2CB, 0, 8, .check_relay = true}, {0x370, 0, 6, .check_relay = true},  // pt bus
                                               {0x184, 2, 8, .check_relay = true}};  // camera bus


  static RxCheck gm_rx_checks[] = {
    GM_COMMON_RX_CHECKS
  };

  static RxCheck gm_ev_rx_checks[] = {
    GM_COMMON_RX_CHECKS
    {.msg = {{0xBD, 0, 7, 40U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  static const CanMsg GM_CAM_TX_MSGS[] = {{0x180, 0, 4, .check_relay = true},  // pt bus
                                          {0x1E1, 2, 7, .check_relay = false}, {0x184, 2, 8, .check_relay = true}};  // camera bus

  gm_hw = GET_FLAG(param, GM_PARAM_HW_CAM) ? GM_CAM : GM_ASCM;

  if (gm_hw == GM_ASCM) {
    gm_long_limits = &GM_ASCM_LONG_LIMITS;
  } else if (gm_hw == GM_CAM) {
    gm_long_limits = &GM_CAM_LONG_LIMITS;
  } else {
  }

  bool gm_cam_long = false;

#ifdef ALLOW_DEBUG
  const uint16_t GM_PARAM_HW_CAM_LONG = 2;
  gm_cam_long = GET_FLAG(param, GM_PARAM_HW_CAM_LONG);
#endif
  gm_pcm_cruise = (gm_hw == GM_CAM) && !gm_cam_long;

  safety_config ret;
  if (gm_hw == GM_CAM) {
    // FIXME: cppcheck thinks that gm_cam_long is always false. This is not true
    // if ALLOW_DEBUG is defined but cppcheck is run without ALLOW_DEBUG
    // cppcheck-suppress knownConditionTrueFalse
    ret = gm_cam_long ? BUILD_SAFETY_CFG(gm_rx_checks, GM_CAM_LONG_TX_MSGS) : BUILD_SAFETY_CFG(gm_rx_checks, GM_CAM_TX_MSGS);
  } else {
    ret = BUILD_SAFETY_CFG(gm_rx_checks, GM_ASCM_TX_MSGS);
  }

  const bool gm_ev = GET_FLAG(param, GM_PARAM_EV);
  if (gm_ev) {
    SET_RX_CHECKS(gm_ev_rx_checks, ret);
  }

  // ASCM does not forward any messages
  if (gm_hw == GM_ASCM) {
    ret.disable_forwarding = true;
  }
  return ret;
}

const safety_hooks gm_hooks = {
  .init = gm_init,
  .rx = gm_rx_hook,
  .tx = gm_tx_hook,
};
