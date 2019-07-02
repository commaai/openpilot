#define CADILLAC_TORQUE_MSG_N 4      // 4 torque messages: 0x151, 0x152, 0x153, 0x154

const int CADILLAC_MAX_STEER = 150; // 1s
// real time torque limit to prevent controls spamming
// the real time limit is 1500/sec
const int CADILLAC_MAX_RT_DELTA = 75;       // max delta torque allowed for real time checks
const uint32_t CADILLAC_RT_INTERVAL = 250000;    // 250ms between real time checks
const int CADILLAC_MAX_RATE_UP = 2;
const int CADILLAC_MAX_RATE_DOWN = 5;
const int CADILLAC_DRIVER_TORQUE_ALLOWANCE = 50;
const int CADILLAC_DRIVER_TORQUE_FACTOR = 4;

int cadillac_ign = 0;
int cadillac_cruise_engaged_last = 0;
int cadillac_rt_torque_last = 0;
const int cadillac_torque_msgs_n = 4;
int cadillac_desired_torque_last[CADILLAC_TORQUE_MSG_N] = {0};
uint32_t cadillac_ts_last = 0;
int cadillac_supercruise_on = 0;
struct sample_t cadillac_torque_driver;         // last few driver torques measured

int cadillac_get_torque_idx(int addr, int array_size) {
  return MIN(MAX(addr - 0x151, 0), array_size);  // 0x151 is id 0, 0x152 is id 1 and so on...
}

static void cadillac_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  if (addr == 356) {
    int torque_driver_new = ((to_push->RDLR & 0x7) << 8) | ((to_push->RDLR >> 8) & 0xFF);
    torque_driver_new = to_signed(torque_driver_new, 11);
    // update array of samples
    update_sample(&cadillac_torque_driver, torque_driver_new);
  }

  // this message isn't all zeros when ignition is on
  if ((addr == 0x160) && (bus == 0)) {
    cadillac_ign = to_push->RDLR > 0;
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  if ((addr == 0x370) && (bus == 0)) {
    int cruise_engaged = to_push->RDLR & 0x800000;  // bit 23
    if (cruise_engaged && !cadillac_cruise_engaged_last) {
      controls_allowed = 1;
    }
    if (!cruise_engaged) {
      controls_allowed = 0;
    }
    cadillac_cruise_engaged_last = cruise_engaged;
  }

  // know supercruise mode and block openpilot msgs if on
  if ((addr == 0x152) || (addr == 0x154)) {
    cadillac_supercruise_on = (to_push->RDHR>>4) & 0x1;
  }
}

static int cadillac_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  int tx = 1;
  int addr = GET_ADDR(to_send);

  // steer cmd checks
  if ((addr == 0x151) || (addr == 0x152) || (addr == 0x153) || (addr == 0x154)) {
    int desired_torque = ((to_send->RDLR & 0x3f) << 8) + ((to_send->RDLR & 0xff00) >> 8);
    int violation = 0;
    uint32_t ts = TIM2->CNT;
    int idx = cadillac_get_torque_idx(addr, CADILLAC_TORQUE_MSG_N);
    desired_torque = to_signed(desired_torque, 14);

    if (controls_allowed) {

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, CADILLAC_MAX_STEER, -CADILLAC_MAX_STEER);

      // *** torque rate limit check ***
      int desired_torque_last = cadillac_desired_torque_last[idx];
      violation |= driver_limit_check(desired_torque, desired_torque_last, &cadillac_torque_driver,
        CADILLAC_MAX_STEER, CADILLAC_MAX_RATE_UP, CADILLAC_MAX_RATE_DOWN,
        CADILLAC_DRIVER_TORQUE_ALLOWANCE, CADILLAC_DRIVER_TORQUE_FACTOR);

      // used next time
      cadillac_desired_torque_last[idx] = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, cadillac_rt_torque_last, CADILLAC_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, cadillac_ts_last);
      if (ts_elapsed > CADILLAC_RT_INTERVAL) {
        cadillac_rt_torque_last = desired_torque;
        cadillac_ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!controls_allowed && (desired_torque != 0)) {
      violation = 1;
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (violation || !controls_allowed) {
      cadillac_desired_torque_last[idx] = 0;
      cadillac_rt_torque_last = 0;
      cadillac_ts_last = ts;
    }

    if (violation || cadillac_supercruise_on) {
      tx = 0;
    }

  }
  return tx;
}

static void cadillac_init(int16_t param) {
  UNUSED(param);
  controls_allowed = 0;
  cadillac_ign = 0;
}

static int cadillac_ign_hook(void) {
  return cadillac_ign;
}

const safety_hooks cadillac_hooks = {
  .init = cadillac_init,
  .rx = cadillac_rx_hook,
  .tx = cadillac_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = cadillac_ign_hook,
  .fwd = default_fwd_hook,
};
