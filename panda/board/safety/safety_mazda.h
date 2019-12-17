
// CAN msgs we care about
#define MAZDA_LKAS 0x243
#define MAZDA_LANEINFO 0x440
#define MAZDA_CRZ_CTRL 0x21c
#define MAZDA_WHEEL_SPEED 0x215
#define MAZDA_STEER_TORQUE 0x240

// CAN bus numbers
#define MAZDA_MAIN 0
#define MAZDA_AUX 1
#define MAZDA_CAM 2

#define MAZDA_MAX_STEER 2048

// max delta torque allowed for real time checks
#define MAZDA_MAX_RT_DELTA 940
// 250ms between real time checks
#define MAZDA_RT_INTERVAL 250000
#define MAZDA_MAX_RATE_UP 10
#define MAZDA_MAX_RATE_DOWN 25
#define MAZDA_DRIVER_TORQUE_ALLOWANCE 15
#define MAZDA_DRIVER_TORQUE_FACTOR 1


int mazda_cruise_engaged_last = 0;
int mazda_rt_torque_last = 0;
int mazda_desired_torque_last = 0;
uint32_t mazda_ts_last = 0;
struct sample_t mazda_torque_driver;         // last few driver torques measured

// track msgs coming from OP so that we know what CAM msgs to drop and what to forward
void mazda_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  if ((addr == MAZDA_STEER_TORQUE) && (bus == MAZDA_MAIN)) {
    int torque_driver_new = GET_BYTE(to_push, 0) - 127;
    // update array of samples
    update_sample(&mazda_torque_driver, torque_driver_new);
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  if ((addr == MAZDA_CRZ_CTRL) && (bus == MAZDA_MAIN)) {
    int cruise_engaged = GET_BYTE(to_push, 0) & 8;
    if (cruise_engaged != 0) {
      if (!mazda_cruise_engaged_last) {
        controls_allowed = 1;
      }
    }
    else {
      controls_allowed = 0;
    }
    mazda_cruise_engaged_last = cruise_engaged;
  }

  // if we see wheel speed msgs on MAZDA_CAM bus then relay is closed
  if ((safety_mode_cnt > RELAY_TRNS_TIMEOUT) && (bus == MAZDA_CAM) && (addr == MAZDA_WHEEL_SPEED)) {
    relay_malfunction = true;
  }
}

static int mazda_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if (relay_malfunction) {
    tx = 0;
  }

  // Check if msg is sent on the main BUS
  if (bus == MAZDA_MAIN) {
    // steer cmd checks
    if (addr == MAZDA_LKAS) {
      int desired_torque = (((GET_BYTE(to_send, 0) & 0x0f) << 8) | GET_BYTE(to_send, 1)) - MAZDA_MAX_STEER;
      bool violation = 0;
      uint32_t ts = TIM2->CNT;

      if (controls_allowed) {

        // *** global torque limit check ***
        violation |= max_limit_check(desired_torque, MAZDA_MAX_STEER, -MAZDA_MAX_STEER);

        // *** torque rate limit check ***
        int desired_torque_last = mazda_desired_torque_last;
        violation |= driver_limit_check(desired_torque, desired_torque_last, &mazda_torque_driver,
                                        MAZDA_MAX_STEER, MAZDA_MAX_RATE_UP, MAZDA_MAX_RATE_DOWN,
                                        MAZDA_DRIVER_TORQUE_ALLOWANCE, MAZDA_DRIVER_TORQUE_FACTOR);
        // used next time
        mazda_desired_torque_last = desired_torque;

        // *** torque real time rate limit check ***
        violation |= rt_rate_limit_check(desired_torque, mazda_rt_torque_last, MAZDA_MAX_RT_DELTA);

        // every RT_INTERVAL set the new limits
        uint32_t ts_elapsed = get_ts_elapsed(ts, mazda_ts_last);
        if (ts_elapsed > ((uint32_t) MAZDA_RT_INTERVAL)) {
          mazda_rt_torque_last = desired_torque;
          mazda_ts_last = ts;
        }
      }

      // no torque if controls is not allowed
      if (!controls_allowed && (desired_torque != 0)) {
        violation = 1;
      }

      // reset to 0 if either controls is not allowed or there's a violation
      if (violation || !controls_allowed) {
        mazda_desired_torque_last = 0;
        mazda_rt_torque_last = 0;
        mazda_ts_last = ts;
      }

      if (violation) {
        tx = 0;
      }
    }
  }
  return tx;
}

static int mazda_fwd_hook(int bus, CAN_FIFOMailBox_TypeDef *to_fwd) {
  int bus_fwd = -1;
  if (!relay_malfunction) {
    int addr = GET_ADDR(to_fwd);
    if (bus == MAZDA_MAIN) {
      bus_fwd = MAZDA_CAM;
    }
    else if (bus == MAZDA_CAM) {
      if (!(addr == MAZDA_LKAS)) {
        bus_fwd = MAZDA_MAIN;
      }
    }
    else {
      bus_fwd = -1;
    }
  }
  return bus_fwd;
}

const safety_hooks mazda_hooks = {
  .init = nooutput_init,
  .rx = mazda_rx_hook,
  .tx = mazda_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = mazda_fwd_hook,
};
