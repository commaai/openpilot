#include "safety_hyundai_common.h"

const SteeringLimits HYUNDAI_CANFD_STEERING_LIMITS = {
  .max_steer = 270,
  .max_rt_delta = 112,
  .max_rt_interval = 250000,
  .max_rate_up = 2,
  .max_rate_down = 3,
  .driver_torque_allowance = 250,
  .driver_torque_factor = 2,
  .type = TorqueDriverLimited,

  // the EPS faults when the steering angle is above a certain threshold for too long. to prevent this,
  // we allow setting torque actuation bit to 0 while maintaining the requested torque value for two consecutive frames
  .min_valid_request_frames = 89,
  .max_invalid_request_frames = 2,
  .min_valid_request_rt_interval = 810000,  // 810ms; a ~10% buffer on cutting every 90 frames
  .has_steer_req_tolerance = true,
};

const CanMsg HYUNDAI_CANFD_HDA2_TX_MSGS[] = {
  {0x50, 0, 16},  // LKAS
  {0x1CF, 1, 8},  // CRUISE_BUTTON
  {0x2A4, 0, 24}, // CAM_0x2A4
};

const CanMsg HYUNDAI_CANFD_HDA2_LONG_TX_MSGS[] = {
  {0x50, 0, 16},  // LKAS
  {0x1CF, 1, 8},  // CRUISE_BUTTON
  {0x2A4, 0, 24}, // CAM_0x2A4
  {0x51, 0, 32},  // ADRV_0x51
  {0x730, 1, 8},  // tester present for ADAS ECU disable
  {0x12A, 1, 16}, // LFA
  {0x160, 1, 16}, // ADRV_0x160
  {0x1E0, 1, 16}, // LFAHDA_CLUSTER
  {0x1A0, 1, 32}, // CRUISE_INFO
  {0x1EA, 1, 32}, // ADRV_0x1ea
  {0x200, 1, 8},  // ADRV_0x200
  {0x345, 1, 8},  // ADRV_0x345
  {0x1DA, 1, 32}, // ADRV_0x1da
};

const CanMsg HYUNDAI_CANFD_HDA1_TX_MSGS[] = {
  {0x12A, 0, 16}, // LFA
  {0x1A0, 0, 32}, // CRUISE_INFO
  {0x1CF, 2, 8},  // CRUISE_BUTTON
  {0x1E0, 0, 16}, // LFAHDA_CLUSTER
};

AddrCheckStruct hyundai_canfd_addr_checks[] = {
  {.msg = {{0x35, 1, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0x35, 0, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0x105, 0, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}}},
  {.msg = {{0x175, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U},
           {0x175, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U}, { 0 }}},
  {.msg = {{0xa0, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0xa0, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }}},
  {.msg = {{0xea, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0xea, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }}},
  {.msg = {{0x1a0, 1, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U},
           {0x1a0, 2, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U}, { 0 }}},
  {.msg = {{0x1cf, 1, 8, .check_checksum = false, .max_counter = 0xfU, .expected_timestep = 20000U},
           {0x1cf, 0, 8, .check_checksum = false, .max_counter = 0xfU, .expected_timestep = 20000U},
           {0x1aa, 0, 16, .check_checksum = false, .max_counter = 0xffU, .expected_timestep = 20000U}}},
};
#define HYUNDAI_CANFD_ADDR_CHECK_LEN (sizeof(hyundai_canfd_addr_checks) / sizeof(hyundai_canfd_addr_checks[0]))

// 0x1a0 is on bus 0
AddrCheckStruct hyundai_canfd_radar_scc_addr_checks[] = {
  {.msg = {{0x35, 1, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0x35, 0, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0x105, 0, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}}},
  {.msg = {{0x175, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U},
           {0x175, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U}, { 0 }}},
  {.msg = {{0xa0, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0xa0, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }}},
  {.msg = {{0xea, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0xea, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }}},
  {.msg = {{0x1a0, 0, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x1cf, 1, 8, .check_checksum = false, .max_counter = 0xfU, .expected_timestep = 20000U},
           {0x1cf, 0, 8, .check_checksum = false, .max_counter = 0xfU, .expected_timestep = 20000U},
           {0x1aa, 0, 16, .check_checksum = false, .max_counter = 0xffU, .expected_timestep = 20000U}}},
};
#define HYUNDAI_CANFD_RADAR_SCC_ADDR_CHECK_LEN (sizeof(hyundai_canfd_radar_scc_addr_checks) / sizeof(hyundai_canfd_radar_scc_addr_checks[0]))

AddrCheckStruct hyundai_canfd_long_addr_checks[] = {
  {.msg = {{0x35, 1, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0x35, 0, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0x105, 0, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}}},
  {.msg = {{0x175, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U},
           {0x175, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U}, { 0 }}},
  {.msg = {{0xa0, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0xa0, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }}},
  {.msg = {{0xea, 1, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U},
           {0xea, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }}},
  {.msg = {{0x1cf, 1, 8, .check_checksum = false, .max_counter = 0xfU, .expected_timestep = 20000U},
           {0x1cf, 0, 8, .check_checksum = false, .max_counter = 0xfU, .expected_timestep = 20000U},
           {0x1aa, 0, 16, .check_checksum = false, .max_counter = 0xffU, .expected_timestep = 20000U}}},
};
#define HYUNDAI_CANFD_LONG_ADDR_CHECK_LEN (sizeof(hyundai_canfd_long_addr_checks) / sizeof(hyundai_canfd_long_addr_checks[0]))

AddrCheckStruct hyundai_canfd_ice_addr_checks[] = {
  {.msg = {{0x100, 0, 32, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0xa0, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0xea, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0x175, 0, 24, .check_checksum = true, .max_counter = 0xffU, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x1aa, 0, 16, .check_checksum = false, .max_counter = 0xffU, .expected_timestep = 20000U}, { 0 }, { 0 }}},
};
#define HYUNDAI_CANFD_ICE_ADDR_CHECK_LEN (sizeof(hyundai_canfd_ice_addr_checks) / sizeof(hyundai_canfd_ice_addr_checks[0]))

addr_checks hyundai_canfd_rx_checks = {hyundai_canfd_addr_checks, HYUNDAI_CANFD_ADDR_CHECK_LEN};


uint16_t hyundai_canfd_crc_lut[256];


const int HYUNDAI_PARAM_CANFD_HDA2 = 16;
const int HYUNDAI_PARAM_CANFD_ALT_BUTTONS = 32;
bool hyundai_canfd_hda2 = false;
bool hyundai_canfd_alt_buttons = false;


static uint8_t hyundai_canfd_get_counter(CANPacket_t *to_push) {
  uint8_t ret = 0;
  if (GET_LEN(to_push) == 8U) {
    ret = GET_BYTE(to_push, 1) >> 4;
  } else {
    ret = GET_BYTE(to_push, 2);
  }
  return ret;
}

static uint32_t hyundai_canfd_get_checksum(CANPacket_t *to_push) {
  uint32_t chksum = GET_BYTE(to_push, 0) | (GET_BYTE(to_push, 1) << 8);
  return chksum;
}

static uint32_t hyundai_canfd_compute_checksum(CANPacket_t *to_push) {
  int len = GET_LEN(to_push);
  uint32_t address = GET_ADDR(to_push);

  uint16_t crc = 0;

  for (int i = 2; i < len; i++) {
    crc = (crc << 8U) ^ hyundai_canfd_crc_lut[(crc >> 8U) ^ GET_BYTE(to_push, i)];
  }

  // Add address to crc
  crc = (crc << 8U) ^ hyundai_canfd_crc_lut[(crc >> 8U) ^ ((address >> 0U) & 0xFFU)];
  crc = (crc << 8U) ^ hyundai_canfd_crc_lut[(crc >> 8U) ^ ((address >> 8U) & 0xFFU)];

  if (len == 8) {
    crc ^= 0x5f29U;
  } else if (len == 16) {
    crc ^= 0x041dU;
  } else if (len == 24) {
    crc ^= 0x819dU;
  } else if (len == 32) {
    crc ^= 0x9f5bU;
  } else {

  }

  return crc;
}

static int hyundai_canfd_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &hyundai_canfd_rx_checks,
                                 hyundai_canfd_get_checksum, hyundai_canfd_compute_checksum, hyundai_canfd_get_counter);

  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  const int pt_bus = hyundai_canfd_hda2 ? 1 : 0;
  const int scc_bus = hyundai_camera_scc ? 2 : pt_bus;

  if (valid && (bus == pt_bus)) {
    // driver torque
    if (addr == 0xea) {
      int torque_driver_new = ((GET_BYTE(to_push, 11) & 0x1fU) << 8U) | GET_BYTE(to_push, 10);
      torque_driver_new -= 4095;
      update_sample(&torque_driver, torque_driver_new);
    }

    // cruise buttons
    const int button_addr = hyundai_canfd_alt_buttons ? 0x1aa : 0x1cf;
    if (addr == button_addr) {
      int main_button = 0;
      int cruise_button = 0;
      if (addr == 0x1cf) {
        cruise_button = GET_BYTE(to_push, 2) & 0x7U;
        main_button = GET_BIT(to_push, 19U);
      } else {
        cruise_button = (GET_BYTE(to_push, 4) >> 4) & 0x7U;
        main_button = GET_BIT(to_push, 34U);
      }
      hyundai_common_cruise_buttons_check(cruise_button, main_button);
    }

    // gas press, different for EV, hybrid, and ICE models
    if ((addr == 0x35) && hyundai_ev_gas_signal) {
      gas_pressed = GET_BYTE(to_push, 5) != 0U;
    } else if ((addr == 0x105) && hyundai_hybrid_gas_signal) {
      gas_pressed = (GET_BIT(to_push, 103U) != 0U) || (GET_BYTE(to_push, 13) != 0U) || (GET_BIT(to_push, 112U) != 0U);
    } else if ((addr == 0x100) && !hyundai_ev_gas_signal && !hyundai_hybrid_gas_signal) {
      gas_pressed = GET_BIT(to_push, 176U) != 0U;
    } else {
    }

    // brake press
    if (addr == 0x175) {
      brake_pressed = GET_BIT(to_push, 81U) != 0U;
    }

    // vehicle moving
    if (addr == 0xa0) {
      uint32_t speed = 0;
      for (int i = 8; i < 15; i+=2) {
        speed += GET_BYTE(to_push, i) | (GET_BYTE(to_push, i + 1) << 8U);
      }
      vehicle_moving = (speed / 4U) > HYUNDAI_STANDSTILL_THRSLD;
    }
  }

  if (valid && (bus == scc_bus)) {
    // cruise state
    if ((addr == 0x1a0) && !hyundai_longitudinal) {
      bool cruise_engaged = ((GET_BYTE(to_push, 8) >> 4) & 0x3U) != 0U;
      hyundai_common_cruise_state_check(cruise_engaged);
    }
  }

  const int steer_addr = hyundai_canfd_hda2 ? 0x50 : 0x12a;
  bool stock_ecu_detected = (addr == steer_addr) && (bus == 0);
  if (hyundai_longitudinal) {
    // on HDA2, ensure ADRV ECU is still knocked out
    // on others, ensure accel msg is blocked from camera
    const int stock_scc_bus = hyundai_canfd_hda2 ? 1 : 0;
    stock_ecu_detected = stock_ecu_detected || ((addr == 0x1a0) && (bus == stock_scc_bus));
  }
  generic_rx_checks(stock_ecu_detected);

  return valid;
}

static int hyundai_canfd_tx_hook(CANPacket_t *to_send) {

  int tx = 0;
  int addr = GET_ADDR(to_send);

  if (hyundai_canfd_hda2 && !hyundai_longitudinal) {
    tx = msg_allowed(to_send, HYUNDAI_CANFD_HDA2_TX_MSGS, sizeof(HYUNDAI_CANFD_HDA2_TX_MSGS)/sizeof(HYUNDAI_CANFD_HDA2_TX_MSGS[0]));
  } else if (hyundai_canfd_hda2 && hyundai_longitudinal) {
    tx = msg_allowed(to_send, HYUNDAI_CANFD_HDA2_LONG_TX_MSGS, sizeof(HYUNDAI_CANFD_HDA2_LONG_TX_MSGS)/sizeof(HYUNDAI_CANFD_HDA2_LONG_TX_MSGS[0]));
  } else {
    tx = msg_allowed(to_send, HYUNDAI_CANFD_HDA1_TX_MSGS, sizeof(HYUNDAI_CANFD_HDA1_TX_MSGS)/sizeof(HYUNDAI_CANFD_HDA1_TX_MSGS[0]));
  }

  // steering
  const int steer_addr = (hyundai_canfd_hda2 && !hyundai_longitudinal) ? 0x50 : 0x12a;
  if (addr == steer_addr) {
    int desired_torque = (((GET_BYTE(to_send, 6) & 0xFU) << 7U) | (GET_BYTE(to_send, 5) >> 1U)) - 1024U;
    bool steer_req = GET_BIT(to_send, 52U) != 0U;

    if (steer_torque_cmd_checks(desired_torque, steer_req, HYUNDAI_CANFD_STEERING_LIMITS)) {
      tx = 0;
    }
  }

  // cruise buttons check
  if (addr == 0x1cf) {
    int button = GET_BYTE(to_send, 2) & 0x7U;
    bool is_cancel = (button == HYUNDAI_BTN_CANCEL);
    bool is_resume = (button == HYUNDAI_BTN_RESUME);

    bool allowed = (is_cancel && cruise_engaged_prev) || (is_resume && controls_allowed);
    if (!allowed) {
      tx = 0;
    }
  }

  // UDS: only tester present ("\x02\x3E\x80\x00\x00\x00\x00\x00") allowed on diagnostics address
  if ((addr == 0x730) && hyundai_canfd_hda2) {
    if ((GET_BYTES_04(to_send) != 0x00803E02U) || (GET_BYTES_48(to_send) != 0x0U)) {
      tx = 0;
    }
  }

  // ACCEL: safety check
  if (addr == 0x1a0) {
    int desired_accel_raw = (((GET_BYTE(to_send, 17) & 0x7U) << 8) | GET_BYTE(to_send, 16)) - 1023U;
    int desired_accel_val = ((GET_BYTE(to_send, 18) << 4) | (GET_BYTE(to_send, 17) >> 4)) - 1023U;

    bool violation = false;

    if (hyundai_longitudinal) {
      violation |= longitudinal_accel_checks(desired_accel_raw, HYUNDAI_LONG_LIMITS);
      violation |= longitudinal_accel_checks(desired_accel_val, HYUNDAI_LONG_LIMITS);
    } else {
      // only used to cancel on here
      if ((desired_accel_raw != 0) || (desired_accel_val != 0)) {
        violation = true;
      }
    }

    if (violation) {
      tx = 0;
    }
  }

  return tx;
}

static int hyundai_canfd_fwd_hook(int bus_num, CANPacket_t *to_fwd) {
  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);

  if (bus_num == 0) {
    bus_fwd = 2;
  }
  if (bus_num == 2) {
    // LKAS for HDA2, LFA for HDA1
    int is_lkas_msg = (((addr == 0x50) || (addr == 0x2a4)) && hyundai_canfd_hda2);
    int is_lfa_msg = ((addr == 0x12a) && !hyundai_canfd_hda2);

    // HUD icons
    int is_lfahda_msg = ((addr == 0x1e0) && !hyundai_canfd_hda2);

    // CRUISE_INFO for non-HDA2, we send our own longitudinal commands
    int is_scc_msg = ((addr == 0x1a0) && hyundai_longitudinal && !hyundai_canfd_hda2);

    int block_msg = is_lkas_msg || is_lfa_msg || is_lfahda_msg || is_scc_msg;
    if (!block_msg) {
      bus_fwd = 0;
    }
  }

  return bus_fwd;
}

static const addr_checks* hyundai_canfd_init(uint16_t param) {
  hyundai_common_init(param);

  gen_crc_lookup_table_16(0x1021, hyundai_canfd_crc_lut);
  hyundai_canfd_hda2 = GET_FLAG(param, HYUNDAI_PARAM_CANFD_HDA2);
  hyundai_canfd_alt_buttons = GET_FLAG(param, HYUNDAI_PARAM_CANFD_ALT_BUTTONS);

  // no long for ICE yet
  if (!hyundai_ev_gas_signal && !hyundai_hybrid_gas_signal) {
    hyundai_longitudinal = false;
  }

  if (hyundai_longitudinal) {
    hyundai_canfd_rx_checks = (addr_checks){hyundai_canfd_long_addr_checks, HYUNDAI_CANFD_LONG_ADDR_CHECK_LEN};
  } else {
    if (!hyundai_ev_gas_signal && !hyundai_hybrid_gas_signal) {
      hyundai_canfd_rx_checks = (addr_checks){hyundai_canfd_ice_addr_checks, HYUNDAI_CANFD_ICE_ADDR_CHECK_LEN};
    } else if (!hyundai_camera_scc && !hyundai_canfd_hda2) {
      hyundai_canfd_rx_checks = (addr_checks){hyundai_canfd_radar_scc_addr_checks, HYUNDAI_CANFD_RADAR_SCC_ADDR_CHECK_LEN};
    } else {
      hyundai_canfd_rx_checks = (addr_checks){hyundai_canfd_addr_checks, HYUNDAI_CANFD_ADDR_CHECK_LEN};
    }
  }

  return &hyundai_canfd_rx_checks;
}

const safety_hooks hyundai_canfd_hooks = {
  .init = hyundai_canfd_init,
  .rx = hyundai_canfd_rx_hook,
  .tx = hyundai_canfd_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = hyundai_canfd_fwd_hook,
};
