struct sample_t torque_meas;           // last 3 motor torques produced by the eps

// global torque limit
const int MAX_TORQUE = 1500;       // max torque cmd allowed ever

// rate based torque limit + stay within actually applied
// packet is sent at 100hz, so this limit is 1000/sec
const int MAX_RATE_UP = 10;        // ramp up slow
const int MAX_RATE_DOWN = 25;      // ramp down fast
const int MAX_TORQUE_ERROR = 350;  // max torque cmd in excess of torque motor

// real time torque limit to prevent controls spamming
// the real time limit is 1500/sec
const int MAX_RT_DELTA = 375;      // max delta torque allowed for real time checks
const int RT_INTERVAL = 250000;    // 250ms between real time checks

// longitudinal limits
const int MAX_ACCEL = 1500;        // 1.5 m/s2
const int MIN_ACCEL = -3000;       // 3.0 m/s2

// global actuation limit state
int actuation_limits = 1;              // by default steer limits are imposed
int dbc_eps_torque_factor = 100;   // conversion factor for STEER_TORQUE_EPS in %: see dbc file

// state of torque limits
int desired_torque_last = 0;       // last desired steer torque
int rt_torque_last = 0;            // last desired torque for real time check
uint32_t ts_last = 0;
int cruise_engaged_last = 0;           // cruise state


static void toyota_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  // get eps motor torque (0.66 factor in dbc)
  if ((to_push->RIR>>21) == 0x260) {
    int torque_meas_new = (((to_push->RDHR) & 0xFF00) | ((to_push->RDHR >> 16) & 0xFF));
    torque_meas_new = to_signed(torque_meas_new, 16);

    // scale by dbc_factor
    torque_meas_new = (torque_meas_new * dbc_eps_torque_factor) / 100;

    // increase torque_meas by 1 to be conservative on rounding
    torque_meas_new += (torque_meas_new > 0 ? 1 : -1);

    // update array of sample
    update_sample(&torque_meas, torque_meas_new);
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  if ((to_push->RIR>>21) == 0x1D2) {
    // 4 bits: 55-52
    int cruise_engaged = to_push->RDHR & 0xF00000;
    if (cruise_engaged && !cruise_engaged_last) {
      controls_allowed = 1;
    } else if (!cruise_engaged) {
      controls_allowed = 0;
    }
    cruise_engaged_last = cruise_engaged;
  }
}

static int toyota_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  // Check if msg is sent on BUS 0
  if (((to_send->RDTR >> 4) & 0xF) == 0) {

    // no IPAS in non IPAS mode
    if (((to_send->RIR>>21) == 0x266) || ((to_send->RIR>>21) == 0x167)) return false;

    // ACCEL: safety check on byte 1-2
    if ((to_send->RIR>>21) == 0x343) {
      int desired_accel = ((to_send->RDLR & 0xFF) << 8) | ((to_send->RDLR >> 8) & 0xFF);
      desired_accel = to_signed(desired_accel, 16);
      if (controls_allowed && actuation_limits) {
        int violation = max_limit_check(desired_accel, MAX_ACCEL, MIN_ACCEL);
        if (violation) return 0;
      } else if (!controls_allowed && (desired_accel != 0)) {
        return 0;
      }
    }

    // STEER: safety check on bytes 2-3
    if ((to_send->RIR>>21) == 0x2E4) {
      int desired_torque = (to_send->RDLR & 0xFF00) | ((to_send->RDLR >> 16) & 0xFF);
      desired_torque = to_signed(desired_torque, 16);
      int violation = 0;

      uint32_t ts = TIM2->CNT;

      // only check if controls are allowed and actuation_limits are imposed
      if (controls_allowed && actuation_limits) {

        // *** global torque limit check ***
        violation |= max_limit_check(desired_torque, MAX_TORQUE, -MAX_TORQUE);

        // *** torque rate limit check ***
        violation |= dist_to_meas_check(desired_torque, desired_torque_last, &torque_meas, MAX_RATE_UP, MAX_RATE_DOWN, MAX_TORQUE_ERROR);

        // used next time
        desired_torque_last = desired_torque;

        // *** torque real time rate limit check ***
        violation |= rt_rate_limit_check(desired_torque, rt_torque_last, MAX_RT_DELTA);

        // every RT_INTERVAL set the new limits
        uint32_t ts_elapsed = get_ts_elapsed(ts, ts_last);
        if (ts_elapsed > RT_INTERVAL) {
          rt_torque_last = desired_torque;
          ts_last = ts;
        }
      }
      
      // no torque if controls is not allowed
      if (!controls_allowed && (desired_torque != 0)) {
        violation = 1;
      }

      // reset to 0 if either controls is not allowed or there's a violation
      if (violation || !controls_allowed) {
        desired_torque_last = 0;
        rt_torque_last = 0;
        ts_last = ts;
      }

      if (violation) {
        return false;
      }
    }
  }

  // 1 allows the message through
  return true;
}

static int toyota_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  // TODO: add safety if using LIN
  return true;
}

static void toyota_init(int16_t param) {
  controls_allowed = 0;
  actuation_limits = 1;
  dbc_eps_torque_factor = param;
}

static int toyota_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  return -1;
}

const safety_hooks toyota_hooks = {
  .init = toyota_init,
  .rx = toyota_rx_hook,
  .tx = toyota_tx_hook,
  .tx_lin = toyota_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = toyota_fwd_hook,
};

static void toyota_nolimits_init(int16_t param) {
  controls_allowed = 0;
  actuation_limits = 0;
  dbc_eps_torque_factor = param;
}

const safety_hooks toyota_nolimits_hooks = {
  .init = toyota_nolimits_init,
  .rx = toyota_rx_hook,
  .tx = toyota_tx_hook,
  .tx_lin = toyota_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = toyota_fwd_hook,
};
