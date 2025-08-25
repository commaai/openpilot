#pragma once

#include "opendbc/safety/safety_declarations.h"

#define PSA_STEERING              757U  // RX from XXX, driver torque
#define PSA_STEERING_ALT          773U  // RX from EPS, steering angle
#define PSA_DYN_CMM               520U  // RX from CMM, gas pedal
#define PSA_HS2_DYN_ABR_38D       909U  // RX from UC_FREIN, speed
#define PSA_HS2_DAT_MDD_CMD_452   1106U // RX from BSI, cruise state
#define PSA_DAT_BSI               1042U // RX from BSI, brake
#define PSA_LANE_KEEP_ASSIST      1010U // TX from OP,  EPS

// CAN bus
#define PSA_MAIN_BUS 0U
#define PSA_ADAS_BUS 1U
#define PSA_CAM_BUS  2U

static uint8_t psa_get_counter(const CANPacket_t *msg) {
  uint8_t cnt = 0;
  if (msg->addr == PSA_HS2_DAT_MDD_CMD_452) {
    cnt = (msg->data[3] >> 4) & 0xFU;
  } else if (msg->addr == PSA_HS2_DYN_ABR_38D) {
    cnt = (msg->data[5] >> 4) & 0xFU;
  } else {
  }
  return cnt;
}

static uint32_t psa_get_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  if (msg->addr == PSA_HS2_DAT_MDD_CMD_452) {
    chksum = msg->data[5] & 0xFU;
  } else if (msg->addr == PSA_HS2_DYN_ABR_38D) {
    chksum = msg->data[5] & 0xFU;
  } else {
  }
  return chksum;
}

static uint8_t _psa_compute_checksum(const CANPacket_t *msg, uint8_t chk_ini, int chk_pos) {
  int len = GET_LEN(msg);

  uint8_t sum = 0;
  for (int i = 0; i < len; i++) {
    uint8_t b = msg->data[i];

    if (i == chk_pos) {
      // set checksum in low nibble to 0
      b &= 0xF0U;
    }
    sum += (b >> 4) + (b & 0xFU);
  }
  return (chk_ini - sum) & 0xFU;
}

static uint32_t psa_compute_checksum(const CANPacket_t *msg) {
  uint8_t chk = 0;
  if (msg->addr == PSA_HS2_DAT_MDD_CMD_452) {
    chk = _psa_compute_checksum(msg, 0x4, 5);
  } else if (msg->addr == PSA_HS2_DYN_ABR_38D) {
    chk = _psa_compute_checksum(msg, 0x7, 5);
  } else {
  }
  return chk;
}

static void psa_rx_hook(const CANPacket_t *msg) {
  if (msg->bus == PSA_MAIN_BUS) {
    if (msg->addr == PSA_DYN_CMM) {
      gas_pressed = msg->data[3] > 0U; // P002_Com_rAPP
    }
    if (msg->addr == PSA_STEERING_ALT) {
      int angle_meas_new = to_signed((msg->data[0] << 8) | msg->data[1], 16); // ANGLE
      update_sample(&angle_meas, angle_meas_new);
    }
    if (msg->addr == PSA_HS2_DYN_ABR_38D) {
      int speed = (msg->data[0] << 8) | msg->data[1];
      vehicle_moving = speed > 0;
      UPDATE_VEHICLE_SPEED(speed * 0.01 * KPH_TO_MS); // VITESSE_VEHICULE_ROUES
    }
  }

  if (msg->bus == PSA_ADAS_BUS) {
    if (msg->addr == PSA_HS2_DAT_MDD_CMD_452) {
      pcm_cruise_check((msg->data[2U] >> 7U) & 1U); // RVV_ACC_ACTIVATION_REQ
    }
  }


  if (msg->bus == PSA_CAM_BUS) {
    if (msg->addr == PSA_DAT_BSI) {
      brake_pressed = (msg->data[0U] >> 5U) & 1U; // P013_MainBrake
    }
  }
}

static bool psa_tx_hook(const CANPacket_t *msg) {
  bool tx = true;
  static const AngleSteeringLimits PSA_STEERING_LIMITS = {
    .max_angle = 3900,
    .angle_deg_to_can = 10,
    .angle_rate_up_lookup = {
      {0., 5., 25.},
      {2.5, 1.5, .2},
    },
    .angle_rate_down_lookup = {
      {0., 5., 25.},
      {5., 2., .3},
    },
  };

  // Safety check for LKA
  if (msg->addr == PSA_LANE_KEEP_ASSIST) {
    // SET_ANGLE
    int desired_angle = to_signed((msg->data[6] << 6) | ((msg->data[7] & 0xFCU) >> 2), 14);
    // TORQUE_FACTOR
    bool lka_active = ((msg->data[5] & 0xFEU) >> 1) == 100U;

    if (steer_angle_cmd_checks(desired_angle, lka_active, PSA_STEERING_LIMITS)) {
      tx = false;
    }
  }
  return tx;
}

static safety_config psa_init(uint16_t param) {
  UNUSED(param);
  static const CanMsg PSA_TX_MSGS[] = {
    {PSA_LANE_KEEP_ASSIST, PSA_MAIN_BUS, 8, .check_relay = true}, // EPS steering
  };

  static RxCheck psa_rx_checks[] = {
    {.msg = {{PSA_HS2_DAT_MDD_CMD_452, PSA_ADAS_BUS, 6, 20U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},                        // cruise state
    {.msg = {{PSA_HS2_DYN_ABR_38D, PSA_MAIN_BUS, 8, 25U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},                            // speed
    {.msg = {{PSA_STEERING_ALT, PSA_MAIN_BUS, 7, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // steering angle
    {.msg = {{PSA_STEERING, PSA_MAIN_BUS, 7, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},     // driver torque
    {.msg = {{PSA_DYN_CMM, PSA_MAIN_BUS, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},      // gas pedal
    {.msg = {{PSA_DAT_BSI, PSA_CAM_BUS, 8, 20U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},        // brake
  };

  return BUILD_SAFETY_CFG(psa_rx_checks, PSA_TX_MSGS);
}

const safety_hooks psa_hooks = {
  .init = psa_init,
  .rx = psa_rx_hook,
  .tx = psa_tx_hook,
  .get_counter = psa_get_counter,
  .get_checksum = psa_get_checksum,
  .compute_checksum = psa_compute_checksum,
};
