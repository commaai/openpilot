const int CHRYSLER_MAX_STEER = 261;
const int CHRYSLER_MAX_RT_DELTA = 112;        // max delta torque allowed for real time checks
const uint32_t CHRYSLER_RT_INTERVAL = 250000;  // 250ms between real time checks
const int CHRYSLER_MAX_RATE_UP = 3;
const int CHRYSLER_MAX_RATE_DOWN = 3;
const int CHRYSLER_MAX_TORQUE_ERROR = 80;    // max torque cmd in excess of torque motor
const AddrBus CHRYSLER_TX_MSGS[] = {{571, 0}, {658, 0}, {678, 0}};

// TODO: do checksum and counter checks
AddrCheckStruct chrysler_rx_checks[] = {
  {.addr = {544}, .bus = 0, .expected_timestep = 10000U},
  {.addr = {500}, .bus = 0, .expected_timestep = 20000U},
};
const int CHRYSLER_RX_CHECK_LEN = sizeof(chrysler_rx_checks) / sizeof(chrysler_rx_checks[0]);

int chrysler_rt_torque_last = 0;
int chrysler_desired_torque_last = 0;
int chrysler_cruise_engaged_last = 0;
uint32_t chrysler_ts_last = 0;
struct sample_t chrysler_torque_meas;         // last few torques measured

static int chrysler_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  // Measured eps torque
  if (addr == 544) {
    int torque_meas_new = ((GET_BYTE(to_push, 4) & 0x7U) << 8) + GET_BYTE(to_push, 5) - 1024U;

    // update array of samples
    update_sample(&chrysler_torque_meas, torque_meas_new);
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  if (addr == 0x1F4) {
    int cruise_engaged = ((GET_BYTE(to_push, 2) & 0x38) >> 3) == 7;
    if (cruise_engaged && !chrysler_cruise_engaged_last) {
      controls_allowed = 1;
    }
    if (!cruise_engaged) {
      controls_allowed = 0;
    }
    chrysler_cruise_engaged_last = cruise_engaged;
  }

  // TODO: add gas pressed check

  // check if stock camera ECU is on bus 0
  if ((safety_mode_cnt > RELAY_TRNS_TIMEOUT) && (bus == 0) && (addr == 0x292)) {
    relay_malfunction = true;
  }
  return 1;
}

static int chrysler_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if (!msg_allowed(addr, bus, CHRYSLER_TX_MSGS, sizeof(CHRYSLER_TX_MSGS) / sizeof(CHRYSLER_TX_MSGS[0]))) {
    tx = 0;
  }

  if (relay_malfunction) {
    tx = 0;
  }

  // LKA STEER
  if (addr == 0x292) {
    int desired_torque = ((GET_BYTE(to_send, 0) & 0x7U) << 8) + GET_BYTE(to_send, 1) - 1024U;
    uint32_t ts = TIM2->CNT;
    bool violation = 0;

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
      tx = 0;
    }
  }

  // FORCE CANCEL: only the cancel button press is allowed
  if (addr == 571) {
    if (GET_BYTE(to_send, 0) != 1) {
      tx = 0;
    }
  }

  return tx;
}

static int chrysler_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {

  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);

  if (!relay_malfunction) {
    // forward CAN 0 -> 2 so stock LKAS camera sees messages
    if (bus_num == 0) {
      bus_fwd = 2;
    }
    // forward all messages from camera except LKAS_COMMAND and LKAS_HUD
    if ((bus_num == 2) && (addr != 658) && (addr != 678)) {
      bus_fwd = 0;
    }
  }
  return bus_fwd;
}


const safety_hooks chrysler_hooks = {
  .init = nooutput_init,
  .rx = chrysler_rx_hook,
  .tx = chrysler_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = chrysler_fwd_hook,
  .addr_check = chrysler_rx_checks,
  .addr_check_len = sizeof(chrysler_rx_checks) / sizeof(chrysler_rx_checks[0]),
};
