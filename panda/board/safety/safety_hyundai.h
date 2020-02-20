const int HYUNDAI_MAX_STEER = 409;             // like stock
const int HYUNDAI_MAX_RT_DELTA = 200;          // max delta torque allowed for real time checks
const uint32_t HYUNDAI_RT_INTERVAL = 250000;    // 250ms between real time checks
const int HYUNDAI_MAX_RATE_UP = 4;
const int HYUNDAI_MAX_RATE_DOWN = 8;
const int HYUNDAI_DRIVER_TORQUE_ALLOWANCE = 50;
const int HYUNDAI_DRIVER_TORQUE_FACTOR = 2;
const AddrBus HYUNDAI_TX_MSGS[] = {{832, 0}, {832, 1}, {1265, 0}, {1265, 1}, {1265, 2}, {593, 2}, {1057, 0}};

// TODO: do checksum and counter checks
AddrCheckStruct hyundai_rx_checks[] = {
  {.addr = {593}, .bus = 0, .expected_timestep = 20000U},
  {.addr = {1057}, .bus = 0, .expected_timestep = 20000U},
};
const int HYUNDAI_RX_CHECK_LEN = sizeof(hyundai_rx_checks) / sizeof(hyundai_rx_checks[0]);

int hyundai_rt_torque_last = 0;
int hyundai_desired_torque_last = 0;
int hyundai_cruise_engaged_last = 0;
uint32_t hyundai_ts_last = 0;
struct sample_t hyundai_torque_driver;         // last few driver torques measured
bool hyundai_has_scc = false;
int OP_LKAS_live = 0;
int OP_MDPS_live = 0;
int OP_CLU_live = 0;
int OP_SCC_live = 0;
int hyundai_mdps_bus = 0;
bool hyundai_LCAN_on_bus1 = false;
bool hyundai_forward_bus1 = false;


static int hyundai_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {

  bool valid = addr_safety_check(to_push, hyundai_rx_checks, HYUNDAI_RX_CHECK_LEN,
                                 NULL, NULL, NULL);

  if (valid) {
    int bus = GET_BUS(to_push);
    int addr = GET_ADDR(to_push);

    if (addr == 593) {
      int torque_driver_new = ((GET_BYTES_04(to_push) & 0x7ff) * 0.79) - 808; // scale down new driver torque signal to match previous one
      // update array of samples
      update_sample(&hyundai_torque_driver, torque_driver_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if (addr == 1057 && OP_SCC_live && (bus != 1 || !hyundai_LCAN_on_bus1)) { // for cars with long control
      hyundai_has_scc = true;
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
    if (addr == 1056 && !OP_SCC_live && (bus != 1 || !hyundai_LCAN_on_bus1)) { // for cars without long control
      hyundai_has_scc = true;
      // 2 bits: 13-14
      int cruise_engaged = GET_BYTES_04(to_push) & 0x1; // ACC main_on signal
      if (cruise_engaged && !hyundai_cruise_engaged_last) {
        controls_allowed = 1;
      }
      if (!cruise_engaged) {
        controls_allowed = 0;
      }
      hyundai_cruise_engaged_last = cruise_engaged;
    }
    // cruise control for car without SCC
    if (addr == 871 && !hyundai_has_scc && OP_SCC_live) {
      // first byte
      int cruise_engaged = (GET_BYTES_04(to_push) & 0xFF);
      if (cruise_engaged && !hyundai_cruise_engaged_last) {
        controls_allowed = 1;
      }
      if (!cruise_engaged) {
        controls_allowed = 0;
      }
      hyundai_cruise_engaged_last = cruise_engaged;
    }
    if (addr == 608 && !hyundai_has_scc && !OP_SCC_live) {
      // bit 25
      int cruise_engaged = (GET_BYTES_04(to_push) >> 25 & 0x1); // ACC main_on signal
      if (cruise_engaged && !hyundai_cruise_engaged_last) {
        controls_allowed = 1;
      }
      if (!cruise_engaged) {
        controls_allowed = 0;
      }
      hyundai_cruise_engaged_last = cruise_engaged;
    }
    // TODO: check gas pressed

    // check if stock camera ECU is on bus 0
    if ((safety_mode_cnt > RELAY_TRNS_TIMEOUT) && (bus == 0) && (addr == 832)) {
      relay_malfunction = true;
    }
    // check if we have a LCAN on Bus1
    if (bus == 1 && (addr == 1296 || addr == 524)) {
      if (hyundai_forward_bus1 || !hyundai_LCAN_on_bus1) {
        hyundai_LCAN_on_bus1 = true;
        hyundai_forward_bus1 = false;
      }
    }
    // check if we have a MDPS on Bus1 and LCAN not on the bus
    if (bus == 1 && (addr == 593 || addr == 897) && !hyundai_LCAN_on_bus1) {
      if (!hyundai_forward_bus1) {
        hyundai_mdps_bus = bus;
        hyundai_forward_bus1 = true;
      }
    }
    // check if we have a SCC on Bus1 and LCAN not on the bus
    if (bus == 1 && addr == 1057 && !hyundai_LCAN_on_bus1) {
      if (!hyundai_forward_bus1) {
        hyundai_forward_bus1 = true;
      }
    }
  }
  return valid;
}

static int hyundai_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if (!msg_allowed(addr, bus, HYUNDAI_TX_MSGS, sizeof(HYUNDAI_TX_MSGS)/sizeof(HYUNDAI_TX_MSGS[0]))) {
    tx = 0;
  }

  if (relay_malfunction) {
    tx = 0;
  }

  // LKA STEER: safety check
  if (addr == 832) {
    OP_LKAS_live = 20;
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
    if (!controls_allowed) { // a reset worsen the issue of Panda blocking some valid LKAS messages
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
  //allow clu11 to be sent to MDPS if MDPS is not on bus0
  if (addr == 1265 && !controls_allowed && (bus != hyundai_mdps_bus || !hyundai_mdps_bus)) { 
    if ((GET_BYTES_04(to_send) & 0x7) != 4) {
      tx = 0;
    }
  }

  if (addr == 593) {OP_MDPS_live = 20;}
  if ((addr == 1265) && (GET_BYTES_04(to_send) & 0x7) == 0) {OP_CLU_live = 20;} // only count non-button msg
  if (addr == 1057) {OP_SCC_live = 20;}

  // 1 allows the message through
  return tx;
}

static int hyundai_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {

  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);
  int fwd_to_bus1 = -1;
  if (hyundai_forward_bus1){fwd_to_bus1 = 1;}

  // forward cam to ccan and viceversa, except lkas cmd
  if (!relay_malfunction) {
    if (bus_num == 0) {
      if (!OP_CLU_live || addr != 1265 || !hyundai_mdps_bus) {
        if (!OP_MDPS_live || addr != 593) {
		  bus_fwd = hyundai_forward_bus1 ? 12 : 2;
		} else {
	      bus_fwd = fwd_to_bus1;  // EON create MDPS for LKAS
		  OP_MDPS_live -= 1;
		}
      } else {
        bus_fwd = 2; // EON create CLU12 for MDPS
		OP_CLU_live -= 1;
      }
    }
    if (bus_num == 1 && hyundai_forward_bus1) {
      if (!OP_MDPS_live || addr != 593) {
        if (!OP_SCC_live || addr != 1057) {
		  bus_fwd = 20;
		} else {
	      bus_fwd = 2;  // EON create SCC12 for Car
		  OP_SCC_live -= 1;
		}
	  } else {
	    bus_fwd = 0;  // EON create MDPS for LKAS
		OP_MDPS_live -= 1;
	  }
    }
    if (bus_num == 2) {
      if (addr != 832 || !OP_LKAS_live) {
        if ((addr != 1057) || (!OP_SCC_live)) {
          bus_fwd = hyundai_forward_bus1 ? 10 : 0;
        } else {
          bus_fwd = fwd_to_bus1;  // EON create SCC12 for Car
		  OP_SCC_live -= 1;
        }
      } else if (!hyundai_mdps_bus) {
		bus_fwd = fwd_to_bus1; // EON create LKAS for Car
        OP_LKAS_live -= 1; 
      } else {
        OP_LKAS_live -= 1; // EON create LKAS for Car and MDPS
      }
    }
  } else {
    if (bus_num == 0) {
      bus_fwd = fwd_to_bus1;
    }
    if (bus_num == 1 && hyundai_forward_bus1) {
      bus_fwd = 0;
    }
  }
  return bus_fwd;
}


const safety_hooks hyundai_hooks = {
  .init = nooutput_init,
  .rx = hyundai_rx_hook,
  .tx = hyundai_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = hyundai_fwd_hook,
  .addr_check = hyundai_rx_checks,
  .addr_check_len = sizeof(hyundai_rx_checks) / sizeof(hyundai_rx_checks[0]),
};
