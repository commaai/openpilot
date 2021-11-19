const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_UP = {
    {2., 7., 17.},
    {5., .8, .25}};

const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_DOWN = {
    {2., 7., 17.},
    {5., 3.5, .8}};

const int TESLA_DEG_TO_CAN = 10;
const float TESLA_MAX_ACCEL = 2.0;  // m/s^2
const float TESLA_MIN_ACCEL = -3.5; // m/s^2

const int TESLA_FLAG_POWERTRAIN = 1;
const int TESLA_FLAG_LONGITUDINAL_CONTROL = 2;

const CanMsg TESLA_TX_MSGS[] = {
  {0x488, 0, 4},  // DAS_steeringControl
  {0x45, 0, 8},   // STW_ACTN_RQ
  {0x45, 2, 8},   // STW_ACTN_RQ
  {0x2b9, 0, 8},  // DAS_control
};
#define TESLA_TX_LEN (sizeof(TESLA_TX_MSGS) / sizeof(TESLA_TX_MSGS[0]))

const CanMsg TESLA_PT_TX_MSGS[] = {
  {0x2bf, 0, 8},  // DAS_control
};
#define TESLA_PT_TX_LEN (sizeof(TESLA_PT_TX_MSGS) / sizeof(TESLA_PT_TX_MSGS[0]))

AddrCheckStruct tesla_addr_checks[] = {
  {.msg = {{0x370, 0, 8, .expected_timestep = 40000U}, { 0 }, { 0 }}},   // EPAS_sysStatus (25Hz)
  {.msg = {{0x108, 0, 8, .expected_timestep = 10000U}, { 0 }, { 0 }}},   // DI_torque1 (100Hz)
  {.msg = {{0x118, 0, 6, .expected_timestep = 10000U}, { 0 }, { 0 }}},   // DI_torque2 (100Hz)
  {.msg = {{0x20a, 0, 8, .expected_timestep = 20000U}, { 0 }, { 0 }}},   // BrakeMessage (50Hz)
  {.msg = {{0x368, 0, 8, .expected_timestep = 100000U}, { 0 }, { 0 }}},  // DI_state (10Hz)
  {.msg = {{0x318, 0, 8, .expected_timestep = 100000U}, { 0 }, { 0 }}},  // GTW_carState (10Hz)
};
#define TESLA_ADDR_CHECK_LEN (sizeof(tesla_addr_checks) / sizeof(tesla_addr_checks[0]))
addr_checks tesla_rx_checks = {tesla_addr_checks, TESLA_ADDR_CHECK_LEN};

AddrCheckStruct tesla_pt_addr_checks[] = {
  {.msg = {{0x106, 0, 8, .expected_timestep = 10000U}, { 0 }, { 0 }}},   // DI_torque1 (100Hz)
  {.msg = {{0x116, 0, 6, .expected_timestep = 10000U}, { 0 }, { 0 }}},   // DI_torque2 (100Hz)
  {.msg = {{0x1f8, 0, 8, .expected_timestep = 20000U}, { 0 }, { 0 }}},   // BrakeMessage (50Hz)
  {.msg = {{0x256, 0, 8, .expected_timestep = 100000U}, { 0 }, { 0 }}},  // DI_state (10Hz)
};
#define TESLA_PT_ADDR_CHECK_LEN (sizeof(tesla_pt_addr_checks) / sizeof(tesla_pt_addr_checks[0]))
addr_checks tesla_pt_rx_checks = {tesla_pt_addr_checks, TESLA_PT_ADDR_CHECK_LEN};

bool tesla_longitudinal = false;
bool tesla_powertrain = false;  // Are we the second panda intercepting the powertrain bus?

static int tesla_rx_hook(CANPacket_t *to_push) {
  bool valid = addr_safety_check(to_push, tesla_powertrain ? (&tesla_pt_rx_checks) : (&tesla_rx_checks),
                                 NULL, NULL, NULL);

  if(valid) {
    int bus = GET_BUS(to_push);
    int addr = GET_ADDR(to_push);

    if(bus == 0) {
      if (!tesla_powertrain) {
        if(addr == 0x370) {
          // Steering angle: (0.1 * val) - 819.2 in deg.
          // Store it 1/10 deg to match steering request
          int angle_meas_new = (((GET_BYTE(to_push, 4) & 0x3F) << 8) | GET_BYTE(to_push, 5)) - 8192;
          update_sample(&angle_meas, angle_meas_new);
        }
      }

      if(addr == (tesla_powertrain ? 0x116 : 0x118)) {
        // Vehicle speed: ((0.05 * val) - 25) * MPH_TO_MPS
        vehicle_speed = (((((GET_BYTE(to_push, 3) & 0x0F) << 8) | (GET_BYTE(to_push, 2))) * 0.05) - 25) * 0.447;
        vehicle_moving = ABS(vehicle_speed) > 0.1;
      }

      if(addr == (tesla_powertrain ? 0x106 : 0x108)) {
        // Gas pressed
        gas_pressed = (GET_BYTE(to_push, 6) != 0);
      }

      if(addr == (tesla_powertrain ? 0x1f8 : 0x20a)) {
        // Brake pressed
        brake_pressed = (((GET_BYTE(to_push, 0) & 0x0C) >> 2) != 1);
      }

      if(addr == (tesla_powertrain ? 0x256 : 0x368)) {
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

    if (tesla_powertrain) {
      // 0x2bf: DAS_control should not be received on bus 0
      generic_rx_checks((addr == 0x2bf) && (bus == 0));
    } else {
      // 0x488: DAS_steeringControl should not be received on bus 0
      generic_rx_checks((addr == 0x488) && (bus == 0));
    }
  }

  return valid;
}


static int tesla_tx_hook(CANPacket_t *to_send) {
  int tx = 1;
  int addr = GET_ADDR(to_send);
  bool violation = false;

  if(!msg_allowed(to_send,
                  tesla_powertrain ? TESLA_PT_TX_MSGS : TESLA_TX_MSGS,
                  tesla_powertrain ? TESLA_PT_TX_LEN : TESLA_TX_LEN)) {
    tx = 0;
  }

  if(!tesla_powertrain && (addr == 0x488)) {
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

  if(!tesla_powertrain && (addr == 0x45)) {
    // No button other than cancel can be sent by us
    int control_lever_status = (GET_BYTE(to_send, 0) & 0x3F);
    if((control_lever_status != 0) && (control_lever_status != 1)) {
      violation = true;
    }
  }

  if(addr == (tesla_powertrain ? 0x2bf : 0x2b9)) {
    // DAS_control: longitudinal control message
    if (tesla_longitudinal) {
      // No AEB events may be sent by openpilot
      int aeb_event = GET_BYTE(to_send, 2) & 0x03;
      if (aeb_event != 0) {
        violation = true;
      }

      // Don't allow any acceleration limits above the safety limits
      int raw_accel_max = ((GET_BYTE(to_send, 6) & 0x1F) << 4) | (GET_BYTE(to_send, 5) >> 4);
      int raw_accel_min = ((GET_BYTE(to_send, 5) & 0x0F) << 5) | (GET_BYTE(to_send, 4) >> 3);
      float accel_max = (0.04 * raw_accel_max) - 15;
      float accel_min = (0.04 * raw_accel_min) - 15;

      if ((accel_max > TESLA_MAX_ACCEL) || (accel_min > TESLA_MAX_ACCEL)){
        violation = true;
      }

      if ((accel_max < TESLA_MIN_ACCEL) || (accel_min < TESLA_MIN_ACCEL)){
        violation = true;
      }
    } else {
      violation = true;
    }
  }

  if(violation) {
    controls_allowed = 0;
    tx = 0;
  }

  return tx;
}

static int tesla_fwd_hook(int bus_num, CANPacket_t *to_fwd) {
  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);

  if(bus_num == 0) {
    // Chassis/PT to autopilot
    bus_fwd = 2;
  }

  if(bus_num == 2) {
    // Autopilot to chassis/PT
    int das_control_addr = (tesla_powertrain ? 0x2bf : 0x2b9);

    bool block_msg = false;
    if (!tesla_powertrain && (addr == 0x488)) {
      block_msg = true;
    }

    if (tesla_longitudinal && (addr == das_control_addr)) {
      block_msg = true;
    }

    if(!block_msg) {
      bus_fwd = 0;
    }
  }

  return bus_fwd;
}

static const addr_checks* tesla_init(int16_t param) {
  tesla_powertrain = GET_FLAG(param, TESLA_FLAG_POWERTRAIN);
  tesla_longitudinal = GET_FLAG(param, TESLA_FLAG_LONGITUDINAL_CONTROL);
  controls_allowed = 0;
  relay_malfunction_reset();

  return tesla_powertrain ? (&tesla_pt_rx_checks) : (&tesla_rx_checks);
}

const safety_hooks tesla_hooks = {
  .init = tesla_init,
  .rx = tesla_rx_hook,
  .tx = tesla_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = tesla_fwd_hook,
};
