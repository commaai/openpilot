const int HYUNDAI_MAX_STEER = 255;             // like stock
const int HYUNDAI_MAX_RT_DELTA = 112;          // max delta torque allowed for real time checks
const uint32_t HYUNDAI_RT_INTERVAL = 250000;    // 250ms between real time checks
const int HYUNDAI_MAX_RATE_UP = 3;
const int HYUNDAI_MAX_RATE_DOWN = 7;
const int HYUNDAI_DRIVER_TORQUE_ALLOWANCE = 50;
const int HYUNDAI_DRIVER_TORQUE_FACTOR = 2;

bool hyundai_camera_detected = 0;
bool hyundai_giraffe_switch_2 = 0;          // is giraffe switch 2 high?
int hyundai_camera_bus = 0;
int hyundai_rt_torque_last = 0;
int hyundai_desired_torque_last = 0;
int hyundai_cruise_engaged_last = 0;
uint32_t hyundai_ts_last = 0;
struct sample_t hyundai_torque_driver;         // last few driver torques measured

static void hyundai_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  if (addr == 897) {
    int torque_driver_new = ((GET_BYTES_04(to_push) >> 11) & 0xfff) - 2048;
    // update array of samples
    update_sample(&hyundai_torque_driver, torque_driver_new);
  }

  // check if stock camera ECU is still online
  if ((bus == 0) && (addr == 832)) {
    hyundai_camera_detected = 1;
    controls_allowed = 0;
  }

  // Find out which bus the camera is on
  if (addr == 832) {
    hyundai_camera_bus = bus;
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  if (addr == 1057) {
    // 2 bits: 13-14
    int cruise_engaged = (GET_BYTES_04(to_push) >> 13) & 0x3;
    if (cruise_engaged && !hyundai_cruise_engaged_last) {
      controls_allowed = 1;
    }
    if (!cruise_engaged) {
      controls_allowed = 0;
    }
    hyundai_cruise_engaged_last = cruise_engaged;
  }

  // 832 is lkas cmd. If it is on camera bus, then giraffe switch 2 is high
  if ((addr == 832) && (bus == hyundai_camera_bus) && (hyundai_camera_bus != 0)) {
    hyundai_giraffe_switch_2 = 1;
  }
}

static int hyundai_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);

  // There can be only one! (camera)
  if (hyundai_camera_detected) {
    tx = 0;
  }

  // LKA STEER: safety check
  if (addr == 832) {
    int desired_torque = ((GET_BYTES_04(to_send) >> 16) & 0x7ff) - 1024;
    uint32_t ts = TIM2->CNT;
    bool violation = 0;

    if (controls_allowed) {

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, HYUNDAI_MAX_STEER, -HYUNDAI_MAX_STEER);

      // *** torque rate limit check ***
      violation |= driver_limit_check(desired_torque, hyundai_desired_torque_last, &hyundai_torque_driver,
        HYUNDAI_MAX_STEER, HYUNDAI_MAX_RATE_UP, HYUNDAI_MAX_RATE_DOWN,
        HYUNDAI_DRIVER_TORQUE_ALLOWANCE, HYUNDAI_DRIVER_TORQUE_FACTOR);

      // used next time
      hyundai_desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, hyundai_rt_torque_last, HYUNDAI_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, hyundai_ts_last);
      if (ts_elapsed > HYUNDAI_RT_INTERVAL) {
        hyundai_rt_torque_last = desired_torque;
        hyundai_ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!controls_allowed && (desired_torque != 0)) {
      violation = 1;
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (violation || !controls_allowed) {
      hyundai_desired_torque_last = 0;
      hyundai_rt_torque_last = 0;
      hyundai_ts_last = ts;
    }

    if (violation) {
      tx = 0;
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button.
  // ensuring that only the cancel button press is sent (VAL 4) when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  // TODO: fix bug preventing the button msg to be fwd'd on bus 2
  //if ((addr == 1265) && !controls_allowed && (bus == 0) {
  //  if ((GET_BYTES_04(to_send) & 0x7) != 4) {
  //    tx = 0;
  //  }
  //}

  // 1 allows the message through
  return tx;
}

static int hyundai_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {

  int bus_fwd = -1;
  // forward cam to ccan and viceversa, except lkas cmd
  if (hyundai_giraffe_switch_2) {
    if (bus_num == 0) {
      bus_fwd = hyundai_camera_bus;
    }
    if (bus_num == hyundai_camera_bus) {
      int addr = GET_ADDR(to_fwd);
      if (addr != 832) {
        bus_fwd = 0;
      }
    }
  }
  return bus_fwd;
}

static void hyundai_init(int16_t param) {
  UNUSED(param);
  controls_allowed = 0;
  hyundai_giraffe_switch_2 = 0;
}

const safety_hooks hyundai_hooks = {
  .init = hyundai_init,
  .rx = hyundai_rx_hook,
  .tx = hyundai_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = hyundai_fwd_hook,
};
