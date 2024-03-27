#include "safety_hyundai_common.h"

#define HYUNDAI_LIMITS(steer, rate_up, rate_down) { \
  .max_steer = (steer), \
  .max_rate_up = (rate_up), \
  .max_rate_down = (rate_down), \
  .max_rt_delta = 112, \
  .max_rt_interval = 250000, \
  .driver_torque_allowance = 50, \
  .driver_torque_factor = 2, \
  .type = TorqueDriverLimited, \
   /* the EPS faults when the steering angle is above a certain threshold for too long. to prevent this, */ \
   /* we allow setting CF_Lkas_ActToi bit to 0 while maintaining the requested torque value for two consecutive frames */ \
  .min_valid_request_frames = 89, \
  .max_invalid_request_frames = 2, \
  .min_valid_request_rt_interval = 810000,  /* 810ms; a ~10% buffer on cutting every 90 frames */ \
  .has_steer_req_tolerance = true, \
}

const SteeringLimits HYUNDAI_STEERING_LIMITS = HYUNDAI_LIMITS(384, 3, 7);
const SteeringLimits HYUNDAI_STEERING_LIMITS_ALT = HYUNDAI_LIMITS(270, 2, 3);

const LongitudinalLimits HYUNDAI_LONG_LIMITS = {
  .max_accel = 200,   // 1/100 m/s2
  .min_accel = -350,  // 1/100 m/s2
};

const CanMsg HYUNDAI_TX_MSGS[] = {
  {0x340, 0, 8}, // LKAS11 Bus 0
  {0x4F1, 0, 4}, // CLU11 Bus 0
  {0x485, 0, 4}, // LFAHDA_MFC Bus 0
};

const CanMsg HYUNDAI_LONG_TX_MSGS[] = {
  {0x340, 0, 8}, // LKAS11 Bus 0
  {0x4F1, 0, 4}, // CLU11 Bus 0
  {0x485, 0, 4}, // LFAHDA_MFC Bus 0
  {0x420, 0, 8}, // SCC11 Bus 0
  {0x421, 0, 8}, // SCC12 Bus 0
  {0x50A, 0, 8}, // SCC13 Bus 0
  {0x389, 0, 8}, // SCC14 Bus 0
  {0x4A2, 0, 2}, // FRT_RADAR11 Bus 0
  {0x38D, 0, 8}, // FCA11 Bus 0
  {0x483, 0, 8}, // FCA12 Bus 0
  {0x7D0, 0, 8}, // radar UDS TX addr Bus 0 (for radar disable)
};

const CanMsg HYUNDAI_CAMERA_SCC_TX_MSGS[] = {
  {0x340, 0, 8}, // LKAS11 Bus 0
  {0x4F1, 2, 4}, // CLU11 Bus 2
  {0x485, 0, 4}, // LFAHDA_MFC Bus 0
};

#define HYUNDAI_COMMON_RX_CHECKS(legacy)                                                                                              \
  {.msg = {{0x260, 0, 8, .check_checksum = true, .max_counter = 3U, .frequency = 100U},                                       \
           {0x371, 0, 8, .frequency = 100U}, { 0 }}},                                                                         \
  {.msg = {{0x386, 0, 8, .check_checksum = !(legacy), .max_counter = (legacy) ? 0U : 15U, .frequency = 100U}, { 0 }, { 0 }}}, \
  {.msg = {{0x394, 0, 8, .check_checksum = !(legacy), .max_counter = (legacy) ? 0U : 7U, .frequency = 100U}, { 0 }, { 0 }}},  \

#define HYUNDAI_SCC12_ADDR_CHECK(scc_bus)                                                                                  \
  {.msg = {{0x421, (scc_bus), 8, .check_checksum = true, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}}, \

RxCheck hyundai_rx_checks[] = {
   HYUNDAI_COMMON_RX_CHECKS(false)
   HYUNDAI_SCC12_ADDR_CHECK(0)
};

RxCheck hyundai_cam_scc_rx_checks[] = {
  HYUNDAI_COMMON_RX_CHECKS(false)
  HYUNDAI_SCC12_ADDR_CHECK(2)
};

RxCheck hyundai_long_rx_checks[] = {
  HYUNDAI_COMMON_RX_CHECKS(false)
  // Use CLU11 (buttons) to manage controls allowed instead of SCC cruise state
  {.msg = {{0x4F1, 0, 4, .check_checksum = false, .max_counter = 15U, .frequency = 50U}, { 0 }, { 0 }}},
};

// older hyundai models have less checks due to missing counters and checksums
RxCheck hyundai_legacy_rx_checks[] = {
  HYUNDAI_COMMON_RX_CHECKS(true)
  HYUNDAI_SCC12_ADDR_CHECK(0)
};

bool hyundai_legacy = false;


static uint8_t hyundai_get_counter(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t cnt = 0;
  if (addr == 0x260) {
    cnt = (GET_BYTE(to_push, 7) >> 4) & 0x3U;
  } else if (addr == 0x386) {
    cnt = ((GET_BYTE(to_push, 3) >> 6) << 2) | (GET_BYTE(to_push, 1) >> 6);
  } else if (addr == 0x394) {
    cnt = (GET_BYTE(to_push, 1) >> 5) & 0x7U;
  } else if (addr == 0x421) {
    cnt = GET_BYTE(to_push, 7) & 0xFU;
  } else if (addr == 0x4F1) {
    cnt = (GET_BYTE(to_push, 3) >> 4) & 0xFU;
  } else {
  }
  return cnt;
}

static uint32_t hyundai_get_checksum(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t chksum = 0;
  if (addr == 0x260) {
    chksum = GET_BYTE(to_push, 7) & 0xFU;
  } else if (addr == 0x386) {
    chksum = ((GET_BYTE(to_push, 7) >> 6) << 2) | (GET_BYTE(to_push, 5) >> 6);
  } else if (addr == 0x394) {
    chksum = GET_BYTE(to_push, 6) & 0xFU;
  } else if (addr == 0x421) {
    chksum = GET_BYTE(to_push, 7) >> 4;
  } else {
  }
  return chksum;
}

static uint32_t hyundai_compute_checksum(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t chksum = 0;
  if (addr == 0x386) {
    // count the bits
    for (int i = 0; i < 8; i++) {
      uint8_t b = GET_BYTE(to_push, i);
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
      if ((addr == 0x394) && (i == 7)) {
        continue; // exclude
      }
      uint8_t b = GET_BYTE(to_push, i);
      if (((addr == 0x260) && (i == 7)) || ((addr == 0x394) && (i == 6)) || ((addr == 0x421) && (i == 7))) {
        b &= (addr == 0x421) ? 0x0FU : 0xF0U; // remove checksum
      }
      chksum += (b % 16U) + (b / 16U);
    }
    chksum = (16U - (chksum %  16U)) % 16U;
  }

  return chksum;
}

static void hyundai_rx_hook(const CANPacket_t *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  // SCC12 is on bus 2 for camera-based SCC cars, bus 0 on all others
  if ((addr == 0x421) && (((bus == 0) && !hyundai_camera_scc) || ((bus == 2) && hyundai_camera_scc))) {
    // 2 bits: 13-14
    int cruise_engaged = (GET_BYTES(to_push, 0, 4) >> 13) & 0x3U;
    hyundai_common_cruise_state_check(cruise_engaged);
  }

  if (bus == 0) {
    if (addr == 0x251) {
      int torque_driver_new = (GET_BYTES(to_push, 0, 2) & 0x7ffU) - 1024U;
      // update array of samples
      update_sample(&torque_driver, torque_driver_new);
    }

    // ACC steering wheel buttons
    if (addr == 0x4F1) {
      int cruise_button = GET_BYTE(to_push, 0) & 0x7U;
      bool main_button = GET_BIT(to_push, 3U);
      hyundai_common_cruise_buttons_check(cruise_button, main_button);
    }

    // gas press, different for EV, hybrid, and ICE models
    if ((addr == 0x371) && hyundai_ev_gas_signal) {
      gas_pressed = (((GET_BYTE(to_push, 4) & 0x7FU) << 1) | GET_BYTE(to_push, 3) >> 7) != 0U;
    } else if ((addr == 0x371) && hyundai_hybrid_gas_signal) {
      gas_pressed = GET_BYTE(to_push, 7) != 0U;
    } else if ((addr == 0x260) && !hyundai_ev_gas_signal && !hyundai_hybrid_gas_signal) {
      gas_pressed = (GET_BYTE(to_push, 7) >> 6) != 0U;
    } else {
    }

    // sample wheel speed, averaging opposite corners
    if (addr == 0x386) {
      uint32_t front_left_speed = GET_BYTES(to_push, 0, 2) & 0x3FFFU;
      uint32_t rear_right_speed = GET_BYTES(to_push, 6, 2) & 0x3FFFU;
      vehicle_moving = (front_left_speed > HYUNDAI_STANDSTILL_THRSLD) || (rear_right_speed > HYUNDAI_STANDSTILL_THRSLD);
    }

    if (addr == 0x394) {
      brake_pressed = ((GET_BYTE(to_push, 5) >> 5U) & 0x3U) == 0x2U;
    }

    bool stock_ecu_detected = (addr == 0x340);

    // If openpilot is controlling longitudinal we need to ensure the radar is turned off
    // Enforce by checking we don't see SCC12
    if (hyundai_longitudinal && (addr == 0x421)) {
      stock_ecu_detected = true;
    }
    generic_rx_checks(stock_ecu_detected);
  }
}

static bool hyundai_tx_hook(const CANPacket_t *to_send) {
  bool tx = true;
  int addr = GET_ADDR(to_send);

  // FCA11: Block any potential actuation
  if (addr == 0x38D) {
    int CR_VSM_DecCmd = GET_BYTE(to_send, 1);
    bool FCA_CmdAct = GET_BIT(to_send, 20U);
    bool CF_VSM_DecCmdAct = GET_BIT(to_send, 31U);

    if ((CR_VSM_DecCmd != 0) || FCA_CmdAct || CF_VSM_DecCmdAct) {
      tx = false;
    }
  }

  // ACCEL: safety check
  if (addr == 0x421) {
    int desired_accel_raw = (((GET_BYTE(to_send, 4) & 0x7U) << 8) | GET_BYTE(to_send, 3)) - 1023U;
    int desired_accel_val = ((GET_BYTE(to_send, 5) << 3) | (GET_BYTE(to_send, 4) >> 5)) - 1023U;

    int aeb_decel_cmd = GET_BYTE(to_send, 2);
    bool aeb_req = GET_BIT(to_send, 54U);

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
  if (addr == 0x340) {
    int desired_torque = ((GET_BYTES(to_send, 0, 4) >> 16) & 0x7ffU) - 1024U;
    bool steer_req = GET_BIT(to_send, 27U);

    const SteeringLimits limits = hyundai_alt_limits ? HYUNDAI_STEERING_LIMITS_ALT : HYUNDAI_STEERING_LIMITS;
    if (steer_torque_cmd_checks(desired_torque, steer_req, limits)) {
      tx = false;
    }
  }

  // UDS: Only tester present ("\x02\x3E\x80\x00\x00\x00\x00\x00") allowed on diagnostics address
  if (addr == 0x7D0) {
    if ((GET_BYTES(to_send, 0, 4) != 0x00803E02U) || (GET_BYTES(to_send, 4, 4) != 0x0U)) {
      tx = false;
    }
  }

  // BUTTONS: used for resume spamming and cruise cancellation
  if ((addr == 0x4F1) && !hyundai_longitudinal) {
    int button = GET_BYTE(to_send, 0) & 0x7U;

    bool allowed_resume = (button == 1) && controls_allowed;
    bool allowed_cancel = (button == 4) && cruise_engaged_prev;
    if (!(allowed_resume || allowed_cancel)) {
      tx = false;
    }
  }

  return tx;
}

static int hyundai_fwd_hook(int bus_num, int addr) {

  int bus_fwd = -1;

  // forward cam to ccan and viceversa, except lkas cmd
  if (bus_num == 0) {
    bus_fwd = 2;
  }
  if ((bus_num == 2) && (addr != 0x340) && (addr != 0x485)) {
    bus_fwd = 0;
  }

  return bus_fwd;
}

static safety_config hyundai_init(uint16_t param) {
  hyundai_common_init(param);
  hyundai_legacy = false;

  if (hyundai_camera_scc) {
    hyundai_longitudinal = false;
  }

  safety_config ret;
  if (hyundai_longitudinal) {
    ret = BUILD_SAFETY_CFG(hyundai_long_rx_checks, HYUNDAI_LONG_TX_MSGS);
  } else if (hyundai_camera_scc) {
    ret = BUILD_SAFETY_CFG(hyundai_cam_scc_rx_checks, HYUNDAI_CAMERA_SCC_TX_MSGS);
  } else {
    ret = BUILD_SAFETY_CFG(hyundai_rx_checks, HYUNDAI_TX_MSGS);
  }
  return ret;
}

static safety_config hyundai_legacy_init(uint16_t param) {
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
  .fwd = hyundai_fwd_hook,
  .get_counter = hyundai_get_counter,
  .get_checksum = hyundai_get_checksum,
  .compute_checksum = hyundai_compute_checksum,
};

const safety_hooks hyundai_legacy_hooks = {
  .init = hyundai_legacy_init,
  .rx = hyundai_rx_hook,
  .tx = hyundai_tx_hook,
  .fwd = hyundai_fwd_hook,
  .get_counter = hyundai_get_counter,
  .get_checksum = hyundai_get_checksum,
  .compute_checksum = hyundai_compute_checksum,
};
