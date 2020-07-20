// CAN msgs we care about
#define MAZDA_LKAS          0x243
#define MAZDA_CRZ_CTRL      0x21c
#define MAZDA_CRZ_BTNS      0x09d
#define MAZDA_STEER_TORQUE  0x240
#define MAZDA_ENGINE_DATA   0x202
#define MAZDA_PEDALS        0x165

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
#define MAZDA_MAX_TORQUE_ERROR 350

// lkas enable speed 52kph, disable at 45kph
#define MAZDA_LKAS_ENABLE_SPEED  5200
#define MAZDA_LKAS_DISABLE_SPEED 4500

const CanMsg MAZDA_TX_MSGS[] = {{MAZDA_LKAS, 0, 8}, {MAZDA_CRZ_BTNS, 0, 8}};
bool mazda_lkas_allowed = false;

AddrCheckStruct mazda_rx_checks[] = {
  {.msg = {{MAZDA_CRZ_CTRL,     0, 8, .expected_timestep = 20000U}}},
  {.msg = {{MAZDA_CRZ_BTNS,     0, 8, .expected_timestep = 100000U}}},
  {.msg = {{MAZDA_STEER_TORQUE, 0, 8, .expected_timestep = 12000U}}},
  {.msg = {{MAZDA_ENGINE_DATA,  0, 8, .expected_timestep = 10000U}}},
  {.msg = {{MAZDA_PEDALS,       0, 8, .expected_timestep = 20000U}}},
};
const int MAZDA_RX_CHECKS_LEN = sizeof(mazda_rx_checks) / sizeof(mazda_rx_checks[0]);

// track msgs coming from OP so that we know what CAM msgs to drop and what to forward
static int mazda_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  bool valid = addr_safety_check(to_push, mazda_rx_checks, MAZDA_RX_CHECKS_LEN,
                            NULL, NULL, NULL);
  if (valid && (GET_BUS(to_push) == MAZDA_MAIN)) {
    int addr = GET_ADDR(to_push);

    if (addr == MAZDA_ENGINE_DATA) {
      // sample speed: scale by 0.01 to get kph
      int speed = (GET_BYTE(to_push, 2) << 8) | GET_BYTE(to_push, 3);

      vehicle_moving = speed > 10; // moving when speed > 0.1 kph

      // Enable LKAS at 52kph going up, disable at 45kph going down
      if (speed > MAZDA_LKAS_ENABLE_SPEED) {
        mazda_lkas_allowed = true;
      } else if (speed < MAZDA_LKAS_DISABLE_SPEED) {
        mazda_lkas_allowed = false;
      } else {
        // Misra-able appeasment block!
      }
    }

    if (addr == MAZDA_STEER_TORQUE) {
      int torque_driver_new = GET_BYTE(to_push, 0) - 127;
      // update array of samples
      update_sample(&torque_driver, torque_driver_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if (addr == MAZDA_CRZ_CTRL) {
      bool cruise_engaged = GET_BYTE(to_push, 0) & 8;
      if (cruise_engaged) {
        if (!cruise_engaged_prev) {
          // do not engage until we hit the speed at which lkas is on
          if (mazda_lkas_allowed) {
            controls_allowed = 1;
          } else {
            controls_allowed = 0;
            cruise_engaged = false;
          }
        }
      } else {
        controls_allowed = 0;
      }
      cruise_engaged_prev = cruise_engaged;
    }

    if (addr == MAZDA_ENGINE_DATA) {
      gas_pressed = (GET_BYTE(to_push, 4) || (GET_BYTE(to_push, 5) & 0xF0));
    }

    if (addr == MAZDA_PEDALS) {
      brake_pressed = (GET_BYTE(to_push, 0) & 0x10);
    }

    generic_rx_checks((addr == MAZDA_LKAS));
  }
  return valid;
}

static int mazda_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if (!msg_allowed(to_send, MAZDA_TX_MSGS, sizeof(MAZDA_TX_MSGS)/sizeof(MAZDA_TX_MSGS[0]))) {
    tx = 0;
  }

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
        violation |= driver_limit_check(desired_torque, desired_torque_last, &torque_driver,
                                        MAZDA_MAX_STEER, MAZDA_MAX_RATE_UP, MAZDA_MAX_RATE_DOWN,
                                        MAZDA_DRIVER_TORQUE_ALLOWANCE, MAZDA_DRIVER_TORQUE_FACTOR);

        // used next time
        desired_torque_last = desired_torque;

        // *** torque real time rate limit check ***
        violation |= rt_rate_limit_check(desired_torque, rt_torque_last, MAZDA_MAX_RT_DELTA);

        // every RT_INTERVAL set the new limits
        uint32_t ts_elapsed = get_ts_elapsed(ts, ts_last);
        if (ts_elapsed > ((uint32_t) MAZDA_RT_INTERVAL)) {
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
    } else if (bus == MAZDA_CAM) {
      if (!(addr == MAZDA_LKAS)) {
        bus_fwd = MAZDA_MAIN;
      }
    } else {
      bus_fwd = -1;
    }
  }
  return bus_fwd;
}

static void mazda_init(int16_t param) {
  UNUSED(param);
  controls_allowed = false;
  relay_malfunction_reset();
  mazda_lkas_allowed = false;
}

const safety_hooks mazda_hooks = {
  .init = mazda_init,
  .rx = mazda_rx_hook,
  .tx = mazda_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = mazda_fwd_hook,
  .addr_check = mazda_rx_checks,
  .addr_check_len = sizeof(mazda_rx_checks) / sizeof(mazda_rx_checks[0]),
};
