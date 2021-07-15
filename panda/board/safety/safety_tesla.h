const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_UP = {
    {2., 7., 17.},
    {5., .8, .25}};

const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_DOWN = {
    {2., 7., 17.},
    {5., 3.5, .8}};

const int TESLA_DEG_TO_CAN = 10;

const CanMsg TESLA_TX_MSGS[] = {
  {0x488, 0, 4},  // DAS_steeringControl
  {0x45, 0, 8},   // STW_ACTN_RQ
  {0x45, 2, 8},   // STW_ACTN_RQ
};

AddrCheckStruct tesla_rx_checks[] = {
  {.msg = {{0x370, 0, 8, .expected_timestep = 40000U}, { 0 }, { 0 }}},   // EPAS_sysStatus (25Hz)
  {.msg = {{0x108, 0, 8, .expected_timestep = 10000U}, { 0 }, { 0 }}},   // DI_torque1 (100Hz)
  {.msg = {{0x118, 0, 6, .expected_timestep = 10000U}, { 0 }, { 0 }}},   // DI_torque2 (100Hz)
  {.msg = {{0x155, 0, 8, .expected_timestep = 20000U}, { 0 }, { 0 }}},   // ESP_B (50Hz)
  {.msg = {{0x20a, 0, 8, .expected_timestep = 20000U}, { 0 }, { 0 }}},   // BrakeMessage (50Hz)
  {.msg = {{0x368, 0, 8, .expected_timestep = 100000U}, { 0 }, { 0 }}},  // DI_state (10Hz)
  {.msg = {{0x318, 0, 8, .expected_timestep = 100000U}, { 0 }, { 0 }}},  // GTW_carState (10Hz)
};
#define TESLA_RX_CHECK_LEN (sizeof(tesla_rx_checks) / sizeof(tesla_rx_checks[0]))

bool autopilot_enabled = false;

static int tesla_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  bool valid = addr_safety_check(to_push, tesla_rx_checks, TESLA_RX_CHECK_LEN,
                                 NULL, NULL, NULL);

  if(valid) {
    int bus = GET_BUS(to_push);
    int addr = GET_ADDR(to_push);

    if(bus == 0) {
      if(addr == 0x370) {
        // Steering angle: (0.1 * val) - 819.2 in deg.
        // Store it 1/10 deg to match steering request
        int angle_meas_new = (((GET_BYTE(to_push, 4) & 0x3F) << 8) | GET_BYTE(to_push, 5)) - 8192;
        update_sample(&angle_meas, angle_meas_new);
      }

      if(addr == 0x155) {
        // Vehicle speed: (0.01 * val) * KPH_TO_MPS
        vehicle_speed = ((GET_BYTE(to_push, 5) << 8) | (GET_BYTE(to_push, 6))) * 0.01 / 3.6;
        vehicle_moving = vehicle_speed > 0.;
      }

      if(addr == 0x108) {
        // Gas pressed
        gas_pressed = (GET_BYTE(to_push, 6) != 0);
      }

      if(addr == 0x20a) {
        // Brake pressed
        brake_pressed = (((GET_BYTE(to_push, 0) & 0x0C) >> 2) != 1);
      }

      if(addr == 0x368) {
        // Cruise state
        int cruise_state = (GET_BYTE(to_push, 1) >> 4);
        bool cruise_engaged = (cruise_state == 2) ||  // ENABLED
                              (cruise_state == 3) ||  // STANDSTILL
                              (cruise_state == 4) ||  // OVERRIDE
                              (cruise_state == 6) ||  // PRE_FAULT
                              (cruise_state == 7);    // PRE_CANCEL

        if(cruise_engaged && !cruise_engaged_prev) {
          controls_allowed = 1;
        }
        if(!cruise_engaged) {
          controls_allowed = 0;
        }
        cruise_engaged_prev = cruise_engaged;
      }
    }

    if (bus == 2) {
      if (addr == 0x399) {
        // Autopilot status
        int autopilot_status = (GET_BYTE(to_push, 0) & 0xF);
        autopilot_enabled = (autopilot_status == 3) ||  // ACTIVE_1
                            (autopilot_status == 4) ||  // ACTIVE_2
                            (autopilot_status == 5);    // ACTIVE_NAVIGATE_ON_AUTOPILOT

        if (autopilot_enabled) {
          controls_allowed = 0;
        }
      }
    }

    // 0x488: DAS_steeringControl should not be received on bus 0
    generic_rx_checks((addr == 0x488) && (bus == 0));
  }

  return valid;
}


static int tesla_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  int tx = 1;
  int addr = GET_ADDR(to_send);
  bool violation = false;

  if(!msg_allowed(to_send, TESLA_TX_MSGS, sizeof(TESLA_TX_MSGS) / sizeof(TESLA_TX_MSGS[0]))) {
    tx = 0;
  }

  if(relay_malfunction) {
    tx = 0;
  }

  if(addr == 0x488) {
    // Steering control: (0.1 * val) - 1638.35 in deg.
    // We use 1/10 deg as a unit here
    int raw_angle_can = (((GET_BYTE(to_send, 0) & 0x7F) << 8) | GET_BYTE(to_send, 1));
    int desired_angle = raw_angle_can - 16384;
    int steer_control_type = GET_BYTE(to_send, 2) >> 6;
    bool steer_control_enabled = (steer_control_type != 0) &&  // NONE
                                 (steer_control_type != 3);    // DISABLED

    // Rate limit while steering
    if(controls_allowed && steer_control_enabled) {
      // Add 1 to not false trigger the violation
      float delta_angle_float;
      delta_angle_float = (interpolate(TESLA_LOOKUP_ANGLE_RATE_UP, vehicle_speed) * TESLA_DEG_TO_CAN);
      int delta_angle_up = (int)(delta_angle_float) + 1;
      delta_angle_float =  (interpolate(TESLA_LOOKUP_ANGLE_RATE_DOWN, vehicle_speed) * TESLA_DEG_TO_CAN);
      int delta_angle_down = (int)(delta_angle_float) + 1;
      int highest_desired_angle = desired_angle_last + ((desired_angle_last > 0) ? delta_angle_up : delta_angle_down);
      int lowest_desired_angle = desired_angle_last - ((desired_angle_last >= 0) ? delta_angle_down : delta_angle_up);

      // Check for violation;
      violation |= max_limit_check(desired_angle, highest_desired_angle, lowest_desired_angle);
    }
    desired_angle_last = desired_angle;

    // Angle should be the same as current angle while not steering
    if(!controls_allowed && ((desired_angle < (angle_meas.min - 1)) || (desired_angle > (angle_meas.max + 1)))) {
      violation = true;
    }

    // No angle control allowed when controls are not allowed
    if(!controls_allowed && steer_control_enabled) {
      violation = true;
    }
  }

  if(addr == 0x45) {
    // No button other than cancel can be sent by us
    int control_lever_status = (GET_BYTE(to_send, 0) & 0x3F);
    if((control_lever_status != 0) && (control_lever_status != 1)) {
      violation = true;
    }
  }

  if(violation) {
    controls_allowed = 0;
    tx = 0;
  }

  return tx;
}

static int tesla_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);

  if(bus_num == 0) {
    // Chassis to autopilot
    bus_fwd = 2;
  }

  if(bus_num == 2) {
    // Autopilot to chassis
    bool block_msg = ((addr == 0x488) && !autopilot_enabled);
    if(!block_msg) {
      bus_fwd = 0;
    }
  }

  if(relay_malfunction) {
    bus_fwd = -1;
  }

  return bus_fwd;
}

static void tesla_init(int16_t param) {
  UNUSED(param);
  controls_allowed = 0;
  relay_malfunction_reset();
}

const safety_hooks tesla_hooks = {
  .init = tesla_init,
  .rx = tesla_rx_hook,
  .tx = tesla_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = tesla_fwd_hook,
  .addr_check = tesla_rx_checks,
  .addr_check_len = TESLA_RX_CHECK_LEN,
};
