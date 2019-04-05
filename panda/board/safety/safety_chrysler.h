const int CHRYSLER_MAX_STEER = 261;
const int CHRYSLER_MAX_RT_DELTA = 112;        // max delta torque allowed for real time checks
const int32_t CHRYSLER_RT_INTERVAL = 250000;  // 250ms between real time checks
const int CHRYSLER_MAX_RATE_UP = 3;
const int CHRYSLER_MAX_RATE_DOWN = 3;
const int CHRYSLER_MAX_TORQUE_ERROR = 80;    // max torque cmd in excess of torque motor

int chrysler_camera_detected = 0;             // is giraffe switch 2 high?
int chrysler_rt_torque_last = 0;
int chrysler_desired_torque_last = 0;
int chrysler_cruise_engaged_last = 0;
uint32_t chrysler_ts_last = 0;
struct sample_t chrysler_torque_meas;         // last few torques measured

static void chrysler_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus = (to_push->RDTR >> 4) & 0xFF;
  uint32_t addr;
  if (to_push->RIR & 4) {
    // Extended
    // Not looked at, but have to be separated
    // to avoid address collision
    addr = to_push->RIR >> 3;
  } else {
    // Normal
    addr = to_push->RIR >> 21;
  }

  // Measured eps torque
  if (addr == 544) {
    int rdhr = to_push->RDHR;
    int torque_meas_new = ((rdhr & 0x7) << 8) + ((rdhr & 0xFF00) >> 8) - 1024;

    // update array of samples
    update_sample(&chrysler_torque_meas, torque_meas_new);
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  if (addr == 0x1f4) {
    int cruise_engaged = ((to_push->RDLR & 0x380000) >> 19) == 7;
    if (cruise_engaged && !chrysler_cruise_engaged_last) {
      controls_allowed = 1;
    } else if (!cruise_engaged) {
      controls_allowed = 0;
    }
    chrysler_cruise_engaged_last = cruise_engaged;
  }

  // check if stock camera ECU is still online
  if (bus == 0 && addr == 0x292) {
    chrysler_camera_detected = 1;
    controls_allowed = 0;
  }
}

static int chrysler_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  // There can be only one! (camera)
  if (chrysler_camera_detected) {
    return 0;
  }

  uint32_t addr;
  if (to_send->RIR & 4) {
    // Extended
    addr = to_send->RIR >> 3;
  } else {
    // Normal
    addr = to_send->RIR >> 21;
  }


  // LKA STEER
  if (addr == 0x292) {
    int rdlr = to_send->RDLR;
    int desired_torque = ((rdlr & 0x7) << 8) + ((rdlr & 0xFF00) >> 8) - 1024;
    uint32_t ts = TIM2->CNT;
    int violation = 0;

    if (controls_allowed) {

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, CHRYSLER_MAX_STEER, -CHRYSLER_MAX_STEER);

      // *** torque rate limit check ***
      violation |= dist_to_meas_check(desired_torque, chrysler_desired_torque_last,
        &chrysler_torque_meas, CHRYSLER_MAX_RATE_UP, CHRYSLER_MAX_RATE_DOWN, CHRYSLER_MAX_TORQUE_ERROR);

      // used next time
      chrysler_desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, chrysler_rt_torque_last, CHRYSLER_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, chrysler_ts_last);
      if (ts_elapsed > CHRYSLER_RT_INTERVAL) {
        chrysler_rt_torque_last = desired_torque;
        chrysler_ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!controls_allowed && (desired_torque != 0)) {
      violation = 1;
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (violation || !controls_allowed) {
      chrysler_desired_torque_last = 0;
      chrysler_rt_torque_last = 0;
      chrysler_ts_last = ts;
    }

    if (violation) {
      return false;
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button.
  // ensuring that only the cancel button press is sent when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  // TODO: fix bug preventing the button msg to be fwd'd on bus 2

  // 1 allows the message through
  return true;
}

static void chrysler_init(int16_t param) {
  chrysler_camera_detected = 0;
}

static int chrysler_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  int32_t addr = to_fwd->RIR >> 21;
  // forward CAN 0 -> 2 so stock LKAS camera sees messages
  if (bus_num == 0 && !chrysler_camera_detected) {
    return 2;
  }
  // forward LKAS_HEARTBIT message from stock camera
  if (bus_num == 2 && !chrysler_camera_detected && addr == 0x2d9) {
    return 0;
  }
  return -1;  // do not forward
}


const safety_hooks chrysler_hooks = {
  .init = chrysler_init,
  .rx = chrysler_rx_hook,
  .tx = chrysler_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = chrysler_fwd_hook,
};
