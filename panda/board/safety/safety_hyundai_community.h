int OP_LKAS_live = 0;
int OP_MDPS_live = 0;
int OP_CLU_live = 0;
int OP_SCC_live = 0;
int car_SCC_live = 0;
int OP_EMS_live = 0;
int HKG_mdps_bus = -1;
int HKG_scc_bus = -1;
const CanMsg HYUNDAI_COMMUNITY_TX_MSGS[] = {
  {832, 0, 8}, {832, 1, 8}, // LKAS11 Bus 0, 1
  {1265, 0, 4}, {1265, 1, 4}, {1265, 2, 4}, // CLU11 Bus 0, 1, 2
  {1157, 0, 4}, // LFAHDA_MFC Bus 0
  {593, 2, 8},  // MDPS12, Bus 2
  {1056, 0, 8}, //   SCC11,  Bus 0
  {1057, 0, 8}, //   SCC12,  Bus 0
  {1290, 0, 8}, //   SCC13,  Bus 0
  {905, 0, 8},  //   SCC14,  Bus 0
  {1186, 0, 8},  //   4a2SCC, Bus 0
  {790, 1, 8}, // EMS11, Bus 1
  {912, 0, 7}, {912,1, 7}, // SPAS11, Bus 0, 1
  {1268, 0, 8}, {1268,1, 8}, // SPAS12, Bus 0, 1
 };

// older hyundai models have less checks due to missing counters and checksums
AddrCheckStruct hyundai_community_rx_checks[] = {
  {.msg = {{608, 0, 8, .check_checksum = true, .max_counter = 3U, .expected_timestep = 10000U},
           {881, 0, 8, .expected_timestep = 10000U}}},
  {.msg = {{902, 0, 8, .expected_timestep = 20000U}}},
  // {.msg = {{916, 0, 8, .expected_timestep = 20000U}}}, some Santa Fe does not have this msg, need to find alternative
};
const int HYUNDAI_COMMUNITY_RX_CHECK_LEN = sizeof(hyundai_community_rx_checks) / sizeof(hyundai_community_rx_checks[0]);

static int hyundai_community_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {

  bool valid;
  int addr = GET_ADDR(to_push);
  int bus = GET_BUS(to_push);

  valid = addr_safety_check(to_push, hyundai_community_rx_checks, HYUNDAI_COMMUNITY_RX_CHECK_LEN,
                            hyundai_get_checksum, hyundai_compute_checksum,
                            hyundai_get_counter);

  if (!valid){
    puts("  CAN RX invalid: "); puth(addr); puts("\n");
  }
  if (bus == 1 && HKG_LCAN_on_bus1) {valid = false;}
  // check if we have a LCAN on Bus1
  if (bus == 1 && (addr == 1296 || addr == 524)) {
    HKG_Lcan_bus1_cnt = 500;
    if (HKG_forward_bus1 || !HKG_LCAN_on_bus1) {
      HKG_LCAN_on_bus1 = true;
      HKG_forward_bus1 = false;
      puts("  LCAN on bus1: forwarding disabled\n");
    }
  }
  // check if LKAS on Bus0
  if (addr == 832) {
    if (bus == 0 && HKG_forward_bus2) {HKG_forward_bus2 = false; HKG_LKAS_bus0_cnt = 20; puts("  LKAS on bus0: forwarding disabled\n");}
    if (bus == 2) {
      if (HKG_LKAS_bus0_cnt > 0) {HKG_LKAS_bus0_cnt--;} else if (!HKG_forward_bus2) {HKG_forward_bus2 = true; puts("  LKAS on bus2 & not on bus0: forwarding enabled\n");}
      if (HKG_Lcan_bus1_cnt > 0) {HKG_Lcan_bus1_cnt--;} else if (HKG_LCAN_on_bus1) {HKG_LCAN_on_bus1 = false; puts("  Lcan not on bus1\n");}
    }
  }
  // check MDPS on Bus
  if ((addr == 593 || addr == 897) && HKG_mdps_bus != bus) {
    if (bus != 1 || (!HKG_LCAN_on_bus1 || HKG_forward_obd)) {
      HKG_mdps_bus = bus;
      if (bus == 1 && !HKG_forward_obd) { puts("  MDPS on bus1\n"); if (!HKG_forward_bus1 && !HKG_LCAN_on_bus1) {HKG_forward_bus1 = true; puts("  bus1 forwarding enabled\n");}}
      else if (bus == 1) {puts("  MDPS on obd bus\n");}
    }
  }
  // check SCC on Bus
  if ((addr == 1056 || addr == 1057) && HKG_scc_bus != bus) {
    if (bus != 1 || !HKG_LCAN_on_bus1) {
      HKG_scc_bus = bus;
      if (bus == 1) { puts("  SCC on bus1\n"); if (!HKG_forward_bus1) {HKG_forward_bus1 = true;puts("  bus1 forwarding enabled\n");}}
      if (bus == 2) { puts("  SCC bus = bus2\n");}
    }
  }

  if (valid) {
    if (addr == 593 && bus == HKG_mdps_bus) {
      int torque_driver_new = ((GET_BYTES_04(to_push) & 0x7ff) * 0.79) - 808; // scale down new driver torque signal to match previous one
      // update array of samples
      update_sample(&torque_driver, torque_driver_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if (addr == 1057 && OP_SCC_live) { // for cars with long control
      car_SCC_live = 50;
      // 2 bits: 13-14
      int cruise_engaged = (GET_BYTES_04(to_push) >> 13) & 0x3;
      if (cruise_engaged && !cruise_engaged_prev) {
        controls_allowed = 1;
        puts("  SCC w/ long control: controls allowed"); puts("\n");
      }
      if (!cruise_engaged) {
        controls_allowed = 0;
        if (controls_allowed) {puts("  SCC w/ long control: controls not allowed"); puts("\n");}
      }
      cruise_engaged_prev = cruise_engaged;
    }
    if (addr == 1056 && !OP_SCC_live) { // for cars without long control
      // 2 bits: 13-14
      int cruise_engaged = GET_BYTES_04(to_push) & 0x1; // ACC main_on signal
      if (cruise_engaged && !cruise_engaged_prev) {
        controls_allowed = 1;
        puts("  SCC w/o long control: controls allowed"); puts("\n");
      }
      if (!cruise_engaged) {
        if (controls_allowed) {puts("  SCC w/o long control: controls not allowed"); puts("\n");}
        controls_allowed = 0;
      }
      cruise_engaged_prev = cruise_engaged;
    }
    // cruise control for car without SCC
    if (addr == 608 && bus == 0 && HKG_scc_bus == -1 && !OP_SCC_live) {
      // bit 25
      int cruise_engaged = (GET_BYTES_04(to_push) >> 25 & 0x1); // ACC main_on signal
      if (cruise_engaged && !cruise_engaged_prev) {
        controls_allowed = 1;
        puts("  non-SCC w/ long control: controls allowed"); puts("\n");
      }
      if (!cruise_engaged) {
        if (controls_allowed) {puts("  non-SCC w/ long control: controls not allowed"); puts("\n");}
        controls_allowed = 0;
      }
      cruise_engaged_prev = cruise_engaged;
    }
    // engage for Cruise control disabled car
    if (addr == 1265 && bus == 0 && OP_SCC_live && !car_SCC_live) {
      // first byte
      int cruise_button = (GET_BYTES_04(to_push) & 0x7);
      // enable on both accel and decel buttons falling edge
      if (!cruise_button && (cruise_engaged_prev == 1 || cruise_engaged_prev == 2)) {
        controls_allowed = 1;
        puts("  non-SCC w/o long control: controls allowed"); puts("\n");
      }
      // disable on cancel rising edge
      if (cruise_button == 4) {
        if (controls_allowed) {puts("  non-SCC w/o long control: controls not allowed"); puts("\n");}
        controls_allowed = 0;
      }
      cruise_engaged_prev = cruise_button;
    }
    // exit controls on rising edge of gas press for cars with long control
    if (addr == 608 && OP_SCC_live && bus == 0) { // EMS16
      gas_pressed = (GET_BYTE(to_push, 7) >> 6) != 0;
    }
    if (addr == 881 && OP_SCC_live && bus == 0) { // E_EMS11
      gas_pressed = (((GET_BYTE(to_push, 4) & 0x7F) << 1) | GET_BYTE(to_push, 3) >> 7) > 5;
    }
    // sample wheel speed, averaging opposite corners
    if (addr == 902 && bus == 0) {
      int hyundai_speed = GET_BYTES_04(to_push) & 0x3FFF;  // FL
      hyundai_speed += (GET_BYTES_48(to_push) >> 16) & 0x3FFF;  // RL
      hyundai_speed /= 2;
      vehicle_moving = hyundai_speed > HYUNDAI_STANDSTILL_THRSLD;
    }
    // exit controls on rising edge of brake press for cars with long control
    if (addr == 916 && OP_SCC_live && bus == 0) {
      brake_pressed = (GET_BYTE(to_push, 6) >> 7) != 0;
    }

    generic_rx_checks((addr == 832 && bus == 0));
  }
  return valid;
}

static int hyundai_community_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if (!msg_allowed(to_send, HYUNDAI_COMMUNITY_TX_MSGS, sizeof(HYUNDAI_COMMUNITY_TX_MSGS)/sizeof(HYUNDAI_COMMUNITY_TX_MSGS[0]))) {
    tx = 0;
    puts("  CAN TX not allowed: "); puth(addr); puts(", "); puth(bus); puts("\n");
  }

  if (relay_malfunction) {
    tx = 0;
    puts("  CAN TX not allowed LKAS on bus0"); puts("\n");
  }

  // LKA STEER: safety check
  if (addr == 832) {
    OP_LKAS_live = 20;
    int desired_torque = ((GET_BYTES_04(to_send) >> 16) & 0x7ff) - 1024;
    uint32_t ts = TIM2->CNT;
    bool violation = 0;

    if (controls_allowed) {

      // *** global torque limit check ***
      bool torque_check = 0;
      violation |= torque_check = max_limit_check(desired_torque, HYUNDAI_MAX_STEER, -HYUNDAI_MAX_STEER);
      if (torque_check) {puts("  LKAS TX not allowed: torque limit check failed!"); puts("\n");}

      // *** torque rate limit check ***
      bool torque_rate_check = 0;
      violation |= torque_rate_check = driver_limit_check(desired_torque, desired_torque_last, &torque_driver,
        HYUNDAI_MAX_STEER, HYUNDAI_MAX_RATE_UP, HYUNDAI_MAX_RATE_DOWN,
        HYUNDAI_DRIVER_TORQUE_ALLOWANCE, HYUNDAI_DRIVER_TORQUE_FACTOR);
      if (torque_rate_check) {puts("  LKAS TX not allowed: torque rate limit check failed!"); puts("\n");}

      // used next time
      desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      bool torque_rt_check = 0;
      violation |= torque_rt_check = rt_rate_limit_check(desired_torque, rt_torque_last, HYUNDAI_MAX_RT_DELTA);
      if (torque_rt_check) {puts("  LKAS TX not allowed: torque real time rate limit check failed!"); puts("\n");}

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, ts_last);
      if (ts_elapsed > HYUNDAI_RT_INTERVAL) {
        rt_torque_last = desired_torque;
        ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!controls_allowed && (desired_torque != 0)) {
      violation = 1;
      puts("  LKAS torque not allowed: controls not allowed!"); puts("\n");
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (!controls_allowed) { // a reset worsen the issue of Panda blocking some valid LKAS messages
      desired_torque_last = 0;
      rt_torque_last = 0;
      ts_last = ts;
    }

    if (violation) {
      tx = 0;
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button.
  // ensuring that only the cancel button press is sent (VAL 4) when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  //allow clu11 to be sent to MDPS if MDPS is not on bus0
  if (addr == 1265 && !controls_allowed && (bus != HKG_mdps_bus && HKG_mdps_bus == 1)) {
    if ((GET_BYTES_04(to_send) & 0x7) != 4) {
      tx = 0;
    }
  }

  if (addr == 593) {OP_MDPS_live = 20;}
  if (addr == 1265 && bus == 1) {OP_CLU_live = 20;} // only count mesage created for MDPS
  if (addr == 1057) {OP_SCC_live = 20; if (car_SCC_live > 0) {car_SCC_live -= 1;}}
  if (addr == 790) {OP_EMS_live = 20;}

  // 1 allows the message through
  return tx;
}

static int hyundai_community_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {

  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);
  int fwd_to_bus1 = -1;
  if (HKG_forward_bus1 || HKG_forward_obd){fwd_to_bus1 = 1;}

  // forward cam to ccan and viceversa, except lkas cmd
  if (HKG_forward_bus2) {
    if (bus_num == 0) {
      if (!OP_CLU_live || addr != 1265 || HKG_mdps_bus == 0) {
        if (!OP_MDPS_live || addr != 593) {
          if (!OP_EMS_live || addr != 790) {
            bus_fwd = fwd_to_bus1 == 1 ? 12 : 2;
          } else {
            bus_fwd = 2;  // EON create EMS11 for MDPS
            OP_EMS_live -= 1;
          }
        } else {
          bus_fwd = fwd_to_bus1;  // EON create MDPS for LKAS
          OP_MDPS_live -= 1;
        }
      } else {
        bus_fwd = 2; // EON create CLU12 for MDPS
        OP_CLU_live -= 1;
      }
    }
    if (bus_num == 1 && (HKG_forward_bus1 || HKG_forward_obd)) {
      if (!OP_MDPS_live || addr != 593) {
        if (!OP_SCC_live || (addr != 1056 && addr != 1057 && addr != 1290 && addr != 905)) {
          bus_fwd = 20;
        } else {
          bus_fwd = 2;  // EON create SCC11 SCC12 SCC13 SCC14 for Car
          OP_SCC_live -= 1;
        }
      } else {
        bus_fwd = 0;  // EON create MDPS for LKAS
        OP_MDPS_live -= 1;
      }
    }
    if (bus_num == 2) {
      if (!OP_LKAS_live || (addr != 832 && addr != 1157)) {
        if (!OP_SCC_live || (addr != 1056 && addr != 1057 && addr != 1290 && addr != 905)) {
          bus_fwd = fwd_to_bus1 == 1 ? 10 : 0;
        } else {
          bus_fwd = fwd_to_bus1;  // EON create SCC12 for Car
          OP_SCC_live -= 1;
        }
      } else if (HKG_mdps_bus == 0) {
        bus_fwd = fwd_to_bus1; // EON create LKAS and LFA for Car
        OP_LKAS_live -= 1;
      } else {
        OP_LKAS_live -= 1; // EON create LKAS and LFA for Car and MDPS
      }
    }
  } else {
    if (bus_num == 0) {
      bus_fwd = fwd_to_bus1;
    }
    if (bus_num == 1 && (HKG_forward_bus1 || HKG_forward_obd)) {
      bus_fwd = 0;
    }
  }
  return bus_fwd;
}

static void hyundai_community_init(int16_t param) {
  UNUSED(param);
  controls_allowed = false;
  relay_malfunction_reset();

  if (board_has_obd() && HKG_forward_obd) {
    current_board->set_can_mode(CAN_MODE_OBD_CAN2);
    puts("  MDPS or SCC on OBD2 CAN: setting can mode obd\n");
  }
}

const safety_hooks hyundai_community_hooks = {
  .init = hyundai_community_init,
  .rx = hyundai_community_rx_hook,
  .tx = hyundai_community_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = hyundai_community_fwd_hook,
  .addr_check = hyundai_community_rx_checks,
  .addr_check_len = sizeof(hyundai_community_rx_checks) / sizeof(hyundai_community_rx_checks[0]),
};
