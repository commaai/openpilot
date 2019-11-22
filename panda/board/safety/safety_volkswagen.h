const int VOLKSWAGEN_MAX_STEER = 250;               // 2.5 Nm (EPS side max of 3.0Nm with fault if violated)
const int VOLKSWAGEN_MAX_RT_DELTA = 75;             // 4 max rate up * 50Hz send rate * 250000 RT interval / 1000000 = 50 ; 50 * 1.5 for safety pad = 75
const uint32_t VOLKSWAGEN_RT_INTERVAL = 250000;     // 250ms between real time checks
const int VOLKSWAGEN_MAX_RATE_UP = 4;               // 2.0 Nm/s available rate of change from the steering rack (EPS side delta-limit of 5.0 Nm/s)
const int VOLKSWAGEN_MAX_RATE_DOWN = 10;            // 5.0 Nm/s available rate of change from the steering rack (EPS side delta-limit of 5.0 Nm/s)
const int VOLKSWAGEN_DRIVER_TORQUE_ALLOWANCE = 80;
const int VOLKSWAGEN_DRIVER_TORQUE_FACTOR = 3;

struct sample_t volkswagen_torque_driver;           // last few driver torques measured
int volkswagen_rt_torque_last = 0;
int volkswagen_desired_torque_last = 0;
uint32_t volkswagen_ts_last = 0;
int volkswagen_gas_prev = 0;

// Safety-relevant CAN messages for the Volkswagen MQB platform.
#define MSG_EPS_01              0x09F
#define MSG_MOTOR_20            0x121
#define MSG_ACC_06              0x122
#define MSG_HCA_01              0x126
#define MSG_GRA_ACC_01          0x12B
#define MSG_LDW_02              0x397
#define MSG_KLEMMEN_STATUS_01   0x3C0

static void volkswagen_init(int16_t param) {
  UNUSED(param); // May use param in the future to indicate MQB vs PQ35/PQ46/NMS vs MLB, or wiring configuration.
  controls_allowed = 0;
}

static void volkswagen_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  // Update driver input torque samples from EPS_01.Driver_Strain for absolute torque, and EPS_01.Driver_Strain_VZ
  // for the direction.
  if ((bus == 0) && (addr == MSG_EPS_01)) {
    int torque_driver_new = GET_BYTE(to_push, 5) | ((GET_BYTE(to_push, 6) & 0x1F) << 8);
    int sign = (GET_BYTE(to_push, 6) & 0x80) >> 7;
    if (sign == 1) {
      torque_driver_new *= -1;
    }

    update_sample(&volkswagen_torque_driver, torque_driver_new);
  }

  // Monitor ACC_06.ACC_Status_ACC for stock ACC status. Because the current MQB port is lateral-only, OP's control
  // allowed state is directly driven by stock ACC engagement. Permit the ACC message to come from either bus, in
  // order to accommodate future camera-side integrations if needed.
  if (addr == MSG_ACC_06) {
    int acc_status = (GET_BYTE(to_push, 7) & 0x70) >> 4;
    controls_allowed = ((acc_status == 3) || (acc_status == 4) || (acc_status == 5)) ? 1 : 0;
  }

  // exit controls on rising edge of gas press. Bits [12-20)
  if (addr == MSG_MOTOR_20) {
    int gas = (GET_BYTES_04(to_push) >> 12) & 0xFF;
    if ((gas > 0) && (volkswagen_gas_prev == 0) && long_controls_allowed) {
      controls_allowed = 0;
    }
    volkswagen_gas_prev = gas;
  }
}

static int volkswagen_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);
  int tx = 1;

  // Safety check for HCA_01 Heading Control Assist torque.
  if (addr == MSG_HCA_01) {
    bool violation = false;

    int desired_torque = GET_BYTE(to_send, 2) | ((GET_BYTE(to_send, 3) & 0x3F) << 8);
    int sign = (GET_BYTE(to_send, 3) & 0x80) >> 7;
    if (sign == 1) {
      desired_torque *= -1;
    }

    uint32_t ts = TIM2->CNT;

    if (controls_allowed) {

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, VOLKSWAGEN_MAX_STEER, -VOLKSWAGEN_MAX_STEER);

      // *** torque rate limit check ***
      violation |= driver_limit_check(desired_torque, volkswagen_desired_torque_last, &volkswagen_torque_driver,
        VOLKSWAGEN_MAX_STEER, VOLKSWAGEN_MAX_RATE_UP, VOLKSWAGEN_MAX_RATE_DOWN,
        VOLKSWAGEN_DRIVER_TORQUE_ALLOWANCE, VOLKSWAGEN_DRIVER_TORQUE_FACTOR);
      volkswagen_desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, volkswagen_rt_torque_last, VOLKSWAGEN_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, volkswagen_ts_last);
      if (ts_elapsed > VOLKSWAGEN_RT_INTERVAL) {
        volkswagen_rt_torque_last = desired_torque;
        volkswagen_ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!controls_allowed && (desired_torque != 0)) {
      violation = true;
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (violation || !controls_allowed) {
      volkswagen_desired_torque_last = 0;
      volkswagen_rt_torque_last = 0;
      volkswagen_ts_last = ts;
    }

    if (violation) {
      tx = 0;
    }
  }

  // FORCE CANCEL: ensuring that only the cancel button press is sent when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  if ((bus == 2) && (addr == MSG_GRA_ACC_01) && !controls_allowed) {
    // disallow resume and set: bits 16 and 19
    if ((GET_BYTE(to_send, 2) & 0x9) != 0) {
      tx = 0;
    }
  }

  // 1 allows the message through
  return tx;
}

static int volkswagen_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  int addr = GET_ADDR(to_fwd);
  int bus_fwd = -1;

  // NOTE: Will need refactoring for other bus layouts, such as no-forwarding at camera or J533 running-gear CAN

  switch (bus_num) {
    case 0:
      // Forward all traffic from J533 gateway to Extended CAN devices.
      bus_fwd = 2;
      break;
    case 2:
      if ((addr == MSG_HCA_01) || (addr == MSG_LDW_02)) {
        // OP takes control of the Heading Control Assist and Lane Departure Warning messages from the camera.
        bus_fwd = -1;
      } else {
        // Forward all remaining traffic from Extended CAN devices to J533 gateway.
        bus_fwd = 0;
      }
      break;
    default:
      // No other buses should be in use; fallback to do-not-forward.
      bus_fwd = -1;
      break;
  }

  return bus_fwd;
}

const safety_hooks volkswagen_hooks = {
  .init = volkswagen_init,
  .rx = volkswagen_rx_hook,
  .tx = volkswagen_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = volkswagen_fwd_hook,
};
