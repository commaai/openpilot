// track the torque measured for limiting
struct sample_t {
  int values[3];
  int min;
  int max;
} sample_t_default = {{0, 0, 0}, 0, 0};
struct sample_t torque_meas;           // last 3 motor torques produced by the eps

// global torque limit
const int32_t MAX_TORQUE = 1500;       // max torque cmd allowed ever

// rate based torque limit + stay within actually applied
// packet is sent at 100hz, so this limit is 1000/sec
const int32_t MAX_RATE_UP = 10;        // ramp up slow
const int32_t MAX_RATE_DOWN = 25;      // ramp down fast
const int32_t MAX_TORQUE_ERROR = 350;  // max torque cmd in excess of torque motor

// real time torque limit to prevent controls spamming
// the real time limit is 1500/sec
const int32_t MAX_RT_DELTA = 375;      // max delta torque allowed for real time checks
const int32_t RT_INTERVAL = 250000;    // 250ms between real time checks

// longitudinal limits
const int16_t MAX_ACCEL = 1500;        // 1.5 m/s2
const int16_t MIN_ACCEL = -3000;       // 3.0 m/s2

// global actuation limit state
int actuation_limits = 1;              // by default steer limits are imposed
int16_t dbc_eps_torque_factor = 100;   // conversion factor for STEER_TORQUE_EPS in %: see dbc file

// state of torque limits
int16_t desired_torque_last = 0;       // last desired steer torque
int16_t rt_torque_last = 0;            // last desired torque for real time check
uint32_t ts_last = 0;
int cruise_engaged_last = 0;           // cruise state

uint32_t get_ts_elapsed(uint32_t ts, uint32_t ts_last) {
  return ts > ts_last ? ts - ts_last : (0xFFFFFFFF - ts_last) + 1 + ts;
}

void update_sample(struct sample_t *sample, int sample_new) {
  for (int i = sizeof(sample->values)/sizeof(sample->values[0]) - 1; i > 0; i--) {
    sample->values[i] = sample->values[i-1];
  }
  sample->values[0] = sample_new;

  // get the minimum and maximum measured torque over the last 3 frames
  sample->min = sample->max = sample->values[0];
  for (int i = 1; i < sizeof(sample->values)/sizeof(sample->values[0]); i++) {
    if (sample->values[i] < sample->min) sample->min = sample->values[i];
    if (sample->values[i] > sample->max) sample->max = sample->values[i];
  }
}

static void toyota_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  // get eps motor torque (0.66 factor in dbc)
  if ((to_push->RIR>>21) == 0x260) {
    int16_t torque_meas_new_16 = (((to_push->RDHR) & 0xFF00) | ((to_push->RDHR >> 16) & 0xFF));

    // increase torque_meas by 1 to be conservative on rounding
    int torque_meas_new = ((int)(torque_meas_new_16) * dbc_eps_torque_factor / 100) + (torque_meas_new_16 > 0 ? 1 : -1);

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
      int16_t desired_accel = ((to_send->RDLR & 0xFF) << 8) | ((to_send->RDLR >> 8) & 0xFF);
      if (controls_allowed && actuation_limits) {
        if ((desired_accel > MAX_ACCEL) || (desired_accel < MIN_ACCEL)) {
          return 0;
        }
      } else if (!controls_allowed && (desired_accel != 0)) {
        return 0;
      }
    }

    // STEER: safety check on bytes 2-3
    if ((to_send->RIR>>21) == 0x2E4) {
      int16_t desired_torque = (to_send->RDLR & 0xFF00) | ((to_send->RDLR >> 16) & 0xFF);
      int16_t violation = 0;

      uint32_t ts = TIM2->CNT;

      // only check if controls are allowed and actuation_limits are imposed
      if (controls_allowed && actuation_limits) {

        // *** global torque limit check ***
        if (desired_torque < -MAX_TORQUE) violation = 1;
        if (desired_torque > MAX_TORQUE) violation = 1;


        // *** torque rate limit check ***
        int16_t highest_allowed_torque = max(desired_torque_last, 0) + MAX_RATE_UP;
        int16_t lowest_allowed_torque = min(desired_torque_last, 0) - MAX_RATE_UP;

        // if we've exceeded the applied torque, we must start moving toward 0
        highest_allowed_torque = min(highest_allowed_torque, max(desired_torque_last - MAX_RATE_DOWN, max(torque_meas.max, 0) + MAX_TORQUE_ERROR));
        lowest_allowed_torque = max(lowest_allowed_torque, min(desired_torque_last + MAX_RATE_DOWN, min(torque_meas.min, 0) - MAX_TORQUE_ERROR));

        // check for violation
        if ((desired_torque < lowest_allowed_torque) || (desired_torque > highest_allowed_torque)) {
          violation = 1;
        }

        // used next time
        desired_torque_last = desired_torque;


        // *** torque real time rate limit check ***
        int16_t highest_rt_torque = max(rt_torque_last, 0) + MAX_RT_DELTA;
        int16_t lowest_rt_torque = min(rt_torque_last, 0) - MAX_RT_DELTA;

        // check for violation
        if ((desired_torque < lowest_rt_torque) || (desired_torque > highest_rt_torque)) {
          violation = 1;
        }

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

static int toyota_ign_hook() {
  return -1;
}

static int toyota_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  return -1;
}

const safety_hooks toyota_hooks = {
  .init = toyota_init,
  .rx = toyota_rx_hook,
  .tx = toyota_tx_hook,
  .tx_lin = toyota_tx_lin_hook,
  .ignition = toyota_ign_hook,
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
  .ignition = toyota_ign_hook,
  .fwd = toyota_fwd_hook,
};
