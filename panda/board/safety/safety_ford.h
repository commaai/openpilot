#pragma once

#include "safety_declarations.h"

// Safety-relevant CAN messages for Ford vehicles.
#define FORD_EngBrakeData          0x165   // RX from PCM, for driver brake pedal and cruise state
#define FORD_EngVehicleSpThrottle  0x204   // RX from PCM, for driver throttle input
#define FORD_DesiredTorqBrk        0x213   // RX from ABS, for standstill state
#define FORD_BrakeSysFeatures      0x415   // RX from ABS, for vehicle speed
#define FORD_EngVehicleSpThrottle2 0x202   // RX from PCM, for second vehicle speed
#define FORD_Yaw_Data_FD1          0x91    // RX from RCM, for yaw rate
#define FORD_Steering_Data_FD1     0x083   // TX by OP, various driver switches and LKAS/CC buttons
#define FORD_ACCDATA               0x186   // TX by OP, ACC controls
#define FORD_ACCDATA_3             0x18A   // TX by OP, ACC/TJA user interface
#define FORD_Lane_Assist_Data1     0x3CA   // TX by OP, Lane Keep Assist
#define FORD_LateralMotionControl  0x3D3   // TX by OP, Lateral Control message
#define FORD_LateralMotionControl2 0x3D6   // TX by OP, alternate Lateral Control message
#define FORD_IPMA_Data             0x3D8   // TX by OP, IPMA and LKAS user interface

// CAN bus numbers.
#define FORD_MAIN_BUS 0U
#define FORD_CAM_BUS  2U

static uint8_t ford_get_counter(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t cnt = 0;
  if (addr == FORD_BrakeSysFeatures) {
    // Signal: VehVActlBrk_No_Cnt
    cnt = (GET_BYTE(to_push, 2) >> 2) & 0xFU;
  } else if (addr == FORD_Yaw_Data_FD1) {
    // Signal: VehRollYaw_No_Cnt
    cnt = GET_BYTE(to_push, 5);
  } else {
  }
  return cnt;
}

static uint32_t ford_get_checksum(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t chksum = 0;
  if (addr == FORD_BrakeSysFeatures) {
    // Signal: VehVActlBrk_No_Cs
    chksum = GET_BYTE(to_push, 3);
  } else if (addr == FORD_Yaw_Data_FD1) {
    // Signal: VehRollYawW_No_Cs
    chksum = GET_BYTE(to_push, 4);
  } else {
  }
  return chksum;
}

static uint32_t ford_compute_checksum(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  uint8_t chksum = 0;
  if (addr == FORD_BrakeSysFeatures) {
    chksum += GET_BYTE(to_push, 0) + GET_BYTE(to_push, 1);  // Veh_V_ActlBrk
    chksum += GET_BYTE(to_push, 2) >> 6;                    // VehVActlBrk_D_Qf
    chksum += (GET_BYTE(to_push, 2) >> 2) & 0xFU;           // VehVActlBrk_No_Cnt
    chksum = 0xFFU - chksum;
  } else if (addr == FORD_Yaw_Data_FD1) {
    chksum += GET_BYTE(to_push, 0) + GET_BYTE(to_push, 1);  // VehRol_W_Actl
    chksum += GET_BYTE(to_push, 2) + GET_BYTE(to_push, 3);  // VehYaw_W_Actl
    chksum += GET_BYTE(to_push, 5);                         // VehRollYaw_No_Cnt
    chksum += GET_BYTE(to_push, 6) >> 6;                    // VehRolWActl_D_Qf
    chksum += (GET_BYTE(to_push, 6) >> 4) & 0x3U;           // VehYawWActl_D_Qf
    chksum = 0xFFU - chksum;
  } else {
  }

  return chksum;
}

static bool ford_get_quality_flag_valid(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);

  bool valid = false;
  if (addr == FORD_BrakeSysFeatures) {
    valid = (GET_BYTE(to_push, 2) >> 6) == 0x3U;           // VehVActlBrk_D_Qf
  } else if (addr == FORD_EngVehicleSpThrottle2) {
    valid = ((GET_BYTE(to_push, 4) >> 5) & 0x3U) == 0x3U;  // VehVActlEng_D_Qf
  } else if (addr == FORD_Yaw_Data_FD1) {
    valid = ((GET_BYTE(to_push, 6) >> 4) & 0x3U) == 0x3U;  // VehYawWActl_D_Qf
  } else {
  }
  return valid;
}

static bool ford_longitudinal = false;

#define FORD_INACTIVE_CURVATURE 1000U
#define FORD_INACTIVE_CURVATURE_RATE 4096U
#define FORD_INACTIVE_PATH_OFFSET 512U
#define FORD_INACTIVE_PATH_ANGLE 1000U

#define FORD_CANFD_INACTIVE_CURVATURE_RATE 1024U

#define FORD_MAX_SPEED_DELTA 2.0  // m/s

static bool ford_lkas_msg_check(int addr) {
  return (addr == FORD_ACCDATA_3)
      || (addr == FORD_Lane_Assist_Data1)
      || (addr == FORD_LateralMotionControl)
      || (addr == FORD_LateralMotionControl2)
      || (addr == FORD_IPMA_Data);
}

// Curvature rate limits
static const SteeringLimits FORD_STEERING_LIMITS = {
  .max_steer = 1000,
  .angle_deg_to_can = 50000,        // 1 / (2e-5) rad to can
  .max_angle_error = 100,           // 0.002 * FORD_STEERING_LIMITS.angle_deg_to_can
  .angle_rate_up_lookup = {
    {5., 25., 25.},
    {0.00045, 0.0001, 0.0001}
  },
  .angle_rate_down_lookup = {
    {5., 25., 25.},
    {0.00045, 0.00015, 0.00015}
  },

  // no blending at low speed due to lack of torque wind-up and inaccurate current curvature
  .angle_error_min_speed = 10.0,    // m/s

  .enforce_angle_error = true,
  .inactive_angle_is_zero = true,
};

static void ford_rx_hook(const CANPacket_t *to_push) {
  if (GET_BUS(to_push) == FORD_MAIN_BUS) {
    int addr = GET_ADDR(to_push);

    // Update in motion state from standstill signal
    if (addr == FORD_DesiredTorqBrk) {
      // Signal: VehStop_D_Stat
      vehicle_moving = ((GET_BYTE(to_push, 3) >> 3) & 0x3U) != 1U;
    }

    // Update vehicle speed
    if (addr == FORD_BrakeSysFeatures) {
      // Signal: Veh_V_ActlBrk
      UPDATE_VEHICLE_SPEED(((GET_BYTE(to_push, 0) << 8) | GET_BYTE(to_push, 1)) * 0.01 / 3.6);
    }

    // Check vehicle speed against a second source
    if (addr == FORD_EngVehicleSpThrottle2) {
      // Disable controls if speeds from ABS and PCM ECUs are too far apart.
      // Signal: Veh_V_ActlEng
      float filtered_pcm_speed = ((GET_BYTE(to_push, 6) << 8) | GET_BYTE(to_push, 7)) * 0.01 / 3.6;
      bool is_invalid_speed = ABS(filtered_pcm_speed - ((float)vehicle_speed.values[0] / VEHICLE_SPEED_FACTOR)) > FORD_MAX_SPEED_DELTA;
      if (is_invalid_speed) {
        controls_allowed = false;
      }
    }

    // Update vehicle yaw rate
    if (addr == FORD_Yaw_Data_FD1) {
      // Signal: VehYaw_W_Actl
      float ford_yaw_rate = (((GET_BYTE(to_push, 2) << 8U) | GET_BYTE(to_push, 3)) * 0.0002) - 6.5;
      float current_curvature = ford_yaw_rate / MAX(vehicle_speed.values[0] / VEHICLE_SPEED_FACTOR, 0.1);
      // convert current curvature into units on CAN for comparison with desired curvature
      update_sample(&angle_meas, ROUND(current_curvature * FORD_STEERING_LIMITS.angle_deg_to_can));
    }

    // Update gas pedal
    if (addr == FORD_EngVehicleSpThrottle) {
      // Pedal position: (0.1 * val) in percent
      // Signal: ApedPos_Pc_ActlArb
      gas_pressed = (((GET_BYTE(to_push, 0) & 0x03U) << 8) | GET_BYTE(to_push, 1)) > 0U;
    }

    // Update brake pedal and cruise state
    if (addr == FORD_EngBrakeData) {
      // Signal: BpedDrvAppl_D_Actl
      brake_pressed = ((GET_BYTE(to_push, 0) >> 4) & 0x3U) == 2U;

      // Signal: CcStat_D_Actl
      unsigned int cruise_state = GET_BYTE(to_push, 1) & 0x07U;
      bool cruise_engaged = (cruise_state == 4U) || (cruise_state == 5U);
      pcm_cruise_check(cruise_engaged);
    }

    // If steering controls messages are received on the destination bus, it's an indication
    // that the relay might be malfunctioning.
    bool stock_ecu_detected = ford_lkas_msg_check(addr);
    if (ford_longitudinal) {
      stock_ecu_detected = stock_ecu_detected || (addr == FORD_ACCDATA);
    }
    generic_rx_checks(stock_ecu_detected);
  }

}

static bool ford_tx_hook(const CANPacket_t *to_send) {
  const LongitudinalLimits FORD_LONG_LIMITS = {
    // acceleration cmd limits (used for brakes)
    // Signal: AccBrkTot_A_Rq
    .max_accel = 5641,       //  1.9999 m/s^s
    .min_accel = 4231,       // -3.4991 m/s^2
    .inactive_accel = 5128,  // -0.0008 m/s^2

    // gas cmd limits
    // Signal: AccPrpl_A_Rq & AccPrpl_A_Pred
    .max_gas = 700,          //  2.0 m/s^2
    .min_gas = 450,          // -0.5 m/s^2
    .inactive_gas = 0,       // -5.0 m/s^2
  };

  bool tx = true;

  int addr = GET_ADDR(to_send);

  // Safety check for ACCDATA accel and brake requests
  if (addr == FORD_ACCDATA) {
    // Signal: AccPrpl_A_Rq
    int gas = ((GET_BYTE(to_send, 6) & 0x3U) << 8) | GET_BYTE(to_send, 7);
    // Signal: AccPrpl_A_Pred
    int gas_pred = ((GET_BYTE(to_send, 2) & 0x3U) << 8) | GET_BYTE(to_send, 3);
    // Signal: AccBrkTot_A_Rq
    int accel = ((GET_BYTE(to_send, 0) & 0x1FU) << 8) | GET_BYTE(to_send, 1);
    // Signal: CmbbDeny_B_Actl
    bool cmbb_deny = GET_BIT(to_send, 37U);

    // Signal: AccBrkPrchg_B_Rq & AccBrkDecel_B_Rq
    bool brake_actuation = GET_BIT(to_send, 54U) || GET_BIT(to_send, 55U);

    bool violation = false;
    violation |= longitudinal_accel_checks(accel, FORD_LONG_LIMITS);
    violation |= longitudinal_gas_checks(gas, FORD_LONG_LIMITS);
    violation |= longitudinal_gas_checks(gas_pred, FORD_LONG_LIMITS);

    // Safety check for stock AEB
    violation |= cmbb_deny; // do not prevent stock AEB actuation

    violation |= !get_longitudinal_allowed() && brake_actuation;

    if (violation) {
      tx = false;
    }
  }

  // Safety check for Steering_Data_FD1 button signals
  // Note: Many other signals in this message are not relevant to safety (e.g. blinkers, wiper switches, high beam)
  // which we passthru in OP.
  if (addr == FORD_Steering_Data_FD1) {
    // Violation if resume button is pressed while controls not allowed, or
    // if cancel button is pressed when cruise isn't engaged.
    bool violation = false;
    violation |= GET_BIT(to_send, 8U) && !cruise_engaged_prev;   // Signal: CcAslButtnCnclPress (cancel)
    violation |= GET_BIT(to_send, 25U) && !controls_allowed;     // Signal: CcAsllButtnResPress (resume)

    if (violation) {
      tx = false;
    }
  }

  // Safety check for Lane_Assist_Data1 action
  if (addr == FORD_Lane_Assist_Data1) {
    // Do not allow steering using Lane_Assist_Data1 (Lane-Departure Aid).
    // This message must be sent for Lane Centering to work, and can include
    // values such as the steering angle or lane curvature for debugging,
    // but the action (LkaActvStats_D2_Req) must be set to zero.
    unsigned int action = GET_BYTE(to_send, 0) >> 5;
    if (action != 0U) {
      tx = false;
    }
  }

  // Safety check for LateralMotionControl action
  if (addr == FORD_LateralMotionControl) {
    // Signal: LatCtl_D_Rq
    bool steer_control_enabled = ((GET_BYTE(to_send, 4) >> 2) & 0x7U) != 0U;
    unsigned int raw_curvature = (GET_BYTE(to_send, 0) << 3) | (GET_BYTE(to_send, 1) >> 5);
    unsigned int raw_curvature_rate = ((GET_BYTE(to_send, 1) & 0x1FU) << 8) | GET_BYTE(to_send, 2);
    unsigned int raw_path_angle = (GET_BYTE(to_send, 3) << 3) | (GET_BYTE(to_send, 4) >> 5);
    unsigned int raw_path_offset = (GET_BYTE(to_send, 5) << 2) | (GET_BYTE(to_send, 6) >> 6);

    // These signals are not yet tested with the current safety limits
    bool violation = (raw_curvature_rate != FORD_INACTIVE_CURVATURE_RATE) || (raw_path_angle != FORD_INACTIVE_PATH_ANGLE) || (raw_path_offset != FORD_INACTIVE_PATH_OFFSET);

    // Check angle error and steer_control_enabled
    int desired_curvature = raw_curvature - FORD_INACTIVE_CURVATURE;  // /FORD_STEERING_LIMITS.angle_deg_to_can to get real curvature
    violation |= steer_angle_cmd_checks(desired_curvature, steer_control_enabled, FORD_STEERING_LIMITS);

    if (violation) {
      tx = false;
    }
  }

  // Safety check for LateralMotionControl2 action
  if (addr == FORD_LateralMotionControl2) {
    // Signal: LatCtl_D2_Rq
    bool steer_control_enabled = ((GET_BYTE(to_send, 0) >> 4) & 0x7U) != 0U;
    unsigned int raw_curvature = (GET_BYTE(to_send, 2) << 3) | (GET_BYTE(to_send, 3) >> 5);
    unsigned int raw_curvature_rate = (GET_BYTE(to_send, 6) << 3) | (GET_BYTE(to_send, 7) >> 5);
    unsigned int raw_path_angle = ((GET_BYTE(to_send, 3) & 0x1FU) << 6) | (GET_BYTE(to_send, 4) >> 2);
    unsigned int raw_path_offset = ((GET_BYTE(to_send, 4) & 0x3U) << 8) | GET_BYTE(to_send, 5);

    // These signals are not yet tested with the current safety limits
    bool violation = (raw_curvature_rate != FORD_CANFD_INACTIVE_CURVATURE_RATE) || (raw_path_angle != FORD_INACTIVE_PATH_ANGLE) || (raw_path_offset != FORD_INACTIVE_PATH_OFFSET);

    // Check angle error and steer_control_enabled
    int desired_curvature = raw_curvature - FORD_INACTIVE_CURVATURE;  // /FORD_STEERING_LIMITS.angle_deg_to_can to get real curvature
    violation |= steer_angle_cmd_checks(desired_curvature, steer_control_enabled, FORD_STEERING_LIMITS);

    if (violation) {
      tx = false;
    }
  }

  return tx;
}

static int ford_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;

  switch (bus_num) {
    case FORD_MAIN_BUS: {
      // Forward all traffic from bus 0 onward
      bus_fwd = FORD_CAM_BUS;
      break;
    }
    case FORD_CAM_BUS: {
      if (ford_lkas_msg_check(addr)) {
        // Block stock LKAS and UI messages
        bus_fwd = -1;
      } else if (ford_longitudinal && (addr == FORD_ACCDATA)) {
        // Block stock ACC message
        bus_fwd = -1;
      } else {
        // Forward remaining traffic
        bus_fwd = FORD_MAIN_BUS;
      }
      break;
    }
    default: {
      // No other buses should be in use; fallback to do-not-forward
      bus_fwd = -1;
      break;
    }
  }

  return bus_fwd;
}

static safety_config ford_init(uint16_t param) {
  bool ford_canfd = false;

  // warning: quality flags are not yet checked in openpilot's CAN parser,
  // this may be the cause of blocked messages
  static RxCheck ford_rx_checks[] = {
    {.msg = {{FORD_BrakeSysFeatures, 0, 8, .check_checksum = true, .max_counter = 15U, .quality_flag=true, .frequency = 50U}, { 0 }, { 0 }}},
    // FORD_EngVehicleSpThrottle2 has a counter that either randomly skips or by 2, likely ECU bug
    // Some hybrid models also experience a bug where this checksum mismatches for one or two frames under heavy acceleration with ACC
    // It has been confirmed that the Bronco Sport's camera only disallows ACC for bad quality flags, not counters or checksums, so we match that
    {.msg = {{FORD_EngVehicleSpThrottle2, 0, 8, .check_checksum = false, .quality_flag=true, .frequency = 50U}, { 0 }, { 0 }}},
    {.msg = {{FORD_Yaw_Data_FD1, 0, 8, .check_checksum = true, .max_counter = 255U, .quality_flag=true, .frequency = 100U}, { 0 }, { 0 }}},
    // These messages have no counter or checksum
    {.msg = {{FORD_EngBrakeData, 0, 8, .frequency = 10U}, { 0 }, { 0 }}},
    {.msg = {{FORD_EngVehicleSpThrottle, 0, 8, .frequency = 100U}, { 0 }, { 0 }}},
    {.msg = {{FORD_DesiredTorqBrk, 0, 8, .frequency = 50U}, { 0 }, { 0 }}},
  };

  static const CanMsg FORD_CANFD_LONG_TX_MSGS[] = {
    {FORD_Steering_Data_FD1, 0, 8},
    {FORD_Steering_Data_FD1, 2, 8},
    {FORD_ACCDATA, 0, 8},
    {FORD_ACCDATA_3, 0, 8},
    {FORD_Lane_Assist_Data1, 0, 8},
    {FORD_LateralMotionControl2, 0, 8},
    {FORD_IPMA_Data, 0, 8},
  };

  static const CanMsg FORD_CANFD_STOCK_TX_MSGS[] = {
    {FORD_Steering_Data_FD1, 0, 8},
    {FORD_Steering_Data_FD1, 2, 8},
    {FORD_ACCDATA_3, 0, 8},
    {FORD_Lane_Assist_Data1, 0, 8},
    {FORD_LateralMotionControl2, 0, 8},
    {FORD_IPMA_Data, 0, 8},
  };

  static const CanMsg FORD_STOCK_TX_MSGS[] = {
    {FORD_Steering_Data_FD1, 0, 8},
    {FORD_Steering_Data_FD1, 2, 8},
    {FORD_ACCDATA_3, 0, 8},
    {FORD_Lane_Assist_Data1, 0, 8},
    {FORD_LateralMotionControl, 0, 8},
    {FORD_IPMA_Data, 0, 8},
  };

  static const CanMsg FORD_LONG_TX_MSGS[] = {
    {FORD_Steering_Data_FD1, 0, 8},
    {FORD_Steering_Data_FD1, 2, 8},
    {FORD_ACCDATA, 0, 8},
    {FORD_ACCDATA_3, 0, 8},
    {FORD_Lane_Assist_Data1, 0, 8},
    {FORD_LateralMotionControl, 0, 8},
    {FORD_IPMA_Data, 0, 8},
  };

  UNUSED(param);
#ifdef ALLOW_DEBUG
  const uint16_t FORD_PARAM_LONGITUDINAL = 1;
  const uint16_t FORD_PARAM_CANFD = 2;
  ford_longitudinal = GET_FLAG(param, FORD_PARAM_LONGITUDINAL);
  ford_canfd = GET_FLAG(param, FORD_PARAM_CANFD);
#endif

  safety_config ret;
  // FIXME: cppcheck thinks that ford_canfd is always false. This is not true
  // if ALLOW_DEBUG is defined but cppcheck is run without ALLOW_DEBUG
  // cppcheck-suppress knownConditionTrueFalse
  if (ford_canfd) {
    ret = ford_longitudinal ? BUILD_SAFETY_CFG(ford_rx_checks, FORD_CANFD_LONG_TX_MSGS) : \
                              BUILD_SAFETY_CFG(ford_rx_checks, FORD_CANFD_STOCK_TX_MSGS);
  } else {
    ret = ford_longitudinal ? BUILD_SAFETY_CFG(ford_rx_checks, FORD_LONG_TX_MSGS) : \
                              BUILD_SAFETY_CFG(ford_rx_checks, FORD_STOCK_TX_MSGS);
  }
  return ret;
}

const safety_hooks ford_hooks = {
  .init = ford_init,
  .rx = ford_rx_hook,
  .tx = ford_tx_hook,
  .fwd = ford_fwd_hook,
  .get_counter = ford_get_counter,
  .get_checksum = ford_get_checksum,
  .compute_checksum = ford_compute_checksum,
  .get_quality_flag_valid = ford_get_quality_flag_valid,
};
