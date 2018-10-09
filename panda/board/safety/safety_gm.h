// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      regen paddle
//      accel rising edge
//      brake rising edge
//      brake > 0mph

const int GM_MAX_STEER = 300;
const int GM_MAX_RT_DELTA = 128;          // max delta torque allowed for real time checks
const int32_t GM_RT_INTERVAL = 250000;    // 250ms between real time checks
const int GM_MAX_RATE_UP = 7;
const int GM_MAX_RATE_DOWN = 17;
const int GM_DRIVER_TORQUE_ALLOWANCE = 50;
const int GM_DRIVER_TORQUE_FACTOR = 4;
const int GM_MAX_GAS = 3072;
const int GM_MAX_REGEN = 1404;
const int GM_MAX_BRAKE = 350;

int gm_brake_prev = 0;
int gm_gas_prev = 0;
int gm_speed = 0;
// silence everything if stock car control ECUs are still online
int gm_ascm_detected = 0;
int gm_ignition_started = 0;
int gm_rt_torque_last = 0;
int gm_desired_torque_last = 0;
uint32_t gm_ts_last = 0;
struct sample_t gm_torque_driver;         // last few driver torques measured

static void gm_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus_number = (to_push->RDTR >> 4) & 0xFF;
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

  if (addr == 388) {
    int torque_driver_new = (((to_push->RDHR >> 16) & 0x7) << 8) | ((to_push->RDHR >> 24) & 0xFF);
    torque_driver_new = to_signed(torque_driver_new, 11);
    // update array of samples
    update_sample(&gm_torque_driver, torque_driver_new);
  }

  if (addr == 0x1f1 && bus_number == 0) {
    //Bit 5 should be ignition "on"
    //Backup plan is Bit 2 (accessory power)
    uint32_t ign = (to_push->RDLR) & 0x20;
    gm_ignition_started = ign > 0;
  }

  // sample speed, really only care if car is moving or not
  // rear left wheel speed
  if (addr == 842) {
    gm_speed = to_push->RDLR & 0xFFFF;
  }

  // Check if ASCM or LKA camera are online
  // on powertrain bus.
  // 384 = ASCMLKASteeringCmd
  // 715 = ASCMGasRegenCmd
  if (bus_number == 0 && (addr == 384 || addr == 715)) {
    gm_ascm_detected = 1;
    controls_allowed = 0;
  }

  // ACC steering wheel buttons
  if (addr == 481) {
    int buttons = (to_push->RDHR >> 12) & 0x7;
    // res/set - enable, cancel button - disable
    if (buttons == 2 || buttons == 3) {
      controls_allowed = 1;
    } else if (buttons == 6) {
      controls_allowed = 0;
    }
  }

  // exit controls on rising edge of brake press or on brake press when
  // speed > 0
  if (addr == 241) {
    int brake = (to_push->RDLR & 0xFF00) >> 8;
    // Brake pedal's potentiometer returns near-zero reading
    // even when pedal is not pressed
    if (brake < 10) {
      brake = 0;
    }
    if (brake && (!gm_brake_prev || gm_speed)) {
       controls_allowed = 0;
    }
    gm_brake_prev = brake;
  }

  // exit controls on rising edge of gas press
  if (addr == 417) {
    int gas = to_push->RDHR & 0xFF0000;
    if (gas && !gm_gas_prev) {
      controls_allowed = 0;
    }
    gm_gas_prev = gas;
  }

  // exit controls on regen paddle
  if (addr == 189) {
    int regen = to_push->RDLR & 0x20;
    if (regen) {
      controls_allowed = 0;
    }
  }
}

// all commands: gas/regen, friction brake and steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int gm_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  // There can be only one! (ASCM)
  if (gm_ascm_detected) {
    return 0;
  }

  // disallow actuator commands if gas or brake (with vehicle moving) are pressed
  // and the the latching controls_allowed flag is True
  int pedal_pressed = gm_gas_prev || (gm_brake_prev && gm_speed);
  int current_controls_allowed = controls_allowed && !pedal_pressed;

  uint32_t addr;
  if (to_send->RIR & 4) {
    // Extended
    addr = to_send->RIR >> 3;
  } else {
    // Normal
    addr = to_send->RIR >> 21;
  }

  // BRAKE: safety check
  if (addr == 789) {
    int rdlr = to_send->RDLR;
    int brake = ((rdlr & 0xF) << 8) + ((rdlr & 0xFF00) >> 8);
    brake = (0x1000 - brake) & 0xFFF;
    if (current_controls_allowed) {
      if (brake > GM_MAX_BRAKE) return 0;
    } else {
      if (brake != 0) return 0;
    }
  }

  // LKA STEER: safety check
  if (addr == 384) {
    int rdlr = to_send->RDLR;
    int desired_torque = ((rdlr & 0x7) << 8) + ((rdlr & 0xFF00) >> 8);
    uint32_t ts = TIM2->CNT;
    int violation = 0;
    desired_torque = to_signed(desired_torque, 11);

    if (current_controls_allowed) {

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, GM_MAX_STEER, -GM_MAX_STEER);

      // *** torque rate limit check ***
      violation |= driver_limit_check(desired_torque, gm_desired_torque_last, &gm_torque_driver,
        GM_MAX_STEER, GM_MAX_RATE_UP, GM_MAX_RATE_DOWN,
        GM_DRIVER_TORQUE_ALLOWANCE, GM_DRIVER_TORQUE_FACTOR);

      // used next time
      gm_desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, gm_rt_torque_last, GM_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, gm_ts_last);
      if (ts_elapsed > GM_RT_INTERVAL) {
        gm_rt_torque_last = desired_torque;
        gm_ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!current_controls_allowed && (desired_torque != 0)) {
      violation = 1;
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (violation || !current_controls_allowed) {
      gm_desired_torque_last = 0;
      gm_rt_torque_last = 0;
      gm_ts_last = ts;
    }

    if (violation) {
      return false;
    }
  }

  // PARK ASSIST STEER: unlimited torque, no thanks
  if (addr == 823) return 0;

  // GAS/REGEN: safety check
  if (addr == 715) {
    int rdlr = to_send->RDLR;
    int gas_regen = ((rdlr & 0x7F0000) >> 11) + ((rdlr & 0xF8000000) >> 27);
    int apply = rdlr & 1;
    if (current_controls_allowed) {
      if (gas_regen > GM_MAX_GAS) return 0;
    } else {
      // Disabled message is !engaed with gas
      // value that corresponds to max regen.
      if (apply || gas_regen != GM_MAX_REGEN) return 0;
    }
  }

  // 1 allows the message through
  return true;
}

static void gm_init(int16_t param) {
  controls_allowed = 0;
  gm_ignition_started = 0;
}

static int gm_ign_hook() {
  return gm_ignition_started;
}

const safety_hooks gm_hooks = {
  .init = gm_init,
  .rx = gm_rx_hook,
  .tx = gm_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = gm_ign_hook,
  .fwd = nooutput_fwd_hook,
};

