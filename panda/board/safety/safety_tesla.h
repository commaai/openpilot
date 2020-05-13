// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      regen paddle
//      accel rising edge
//      brake rising edge
//      brake > 0mph
//
bool fmax_limit_check(float val, const float MAX_VAL, const float MIN_VAL) {
  return (val > MAX_VAL) || (val < MIN_VAL);
}

// 2m/s are added to be less restrictive
const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_UP = {
    {2., 7., 17.},
    {5., .8, .25}};

const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_DOWN = {
    {2., 7., 17.},
    {5., 3.5, .8}};

const struct lookup_t TESLA_LOOKUP_MAX_ANGLE = {
    {2., 29., 38.},
    {410., 92., 36.}};

const uint32_t TESLA_RT_INTERVAL = 250000; // 250ms between real time checks

// state of angle limits
float tesla_desired_angle_last = 0; // last desired steer angle
float tesla_rt_angle_last = 0.; // last real time angle
float tesla_ts_angle_last = 0;

int tesla_controls_allowed_last = 0;

int tesla_speed = 0;
int eac_status = 0;

void set_gmlan_digital_output(int to_set);
void reset_gmlan_switch_timeout(void);
void gmlan_switch_init(int timeout_enable);


static int tesla_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  set_gmlan_digital_output(0); // #define GMLAN_HIGH 0
  reset_gmlan_switch_timeout(); //we're still in tesla safety mode, reset the timeout counter and make sure our output is enabled

  int addr = GET_ADDR(to_push);

  if (addr == 0x45) {
    // 6 bits starting at position 0
    int lever_position = GET_BYTE(to_push, 0) & 0x3F;
    if (lever_position == 2) { // pull forward
      // activate openpilot
      controls_allowed = 1;
    }
    if (lever_position == 1) { // push towards the back
      // deactivate openpilot
      controls_allowed = 0;
    }
  }

  // exit controls on brake press
  // DI_torque2::DI_brakePedal 0x118
  if (addr == 0x118) {
    // 1 bit at position 16
    if ((GET_BYTE(to_push, 1) & 0x80) != 0) {
      // disable break cancel by commenting line below
      controls_allowed = 0;
    }
    //get vehicle speed in m/s. Tesla gives MPH
    tesla_speed = (((((GET_BYTE(to_push, 3) & 0xF) << 8) + GET_BYTE(to_push, 2)) * 0.05) - 25) * 1.609 / 3.6;
    if (tesla_speed < 0) {
      tesla_speed = 0;
    }
  }

  // exit controls on EPAS error
  // EPAS_sysStatus::EPAS_eacStatus 0x370
  if (addr == 0x370) {
    // if EPAS_eacStatus is not 1 or 2, disable control
    eac_status = (GET_BYTE(to_push, 6) >> 5) & 0x7;
    // For human steering override we must not disable controls when eac_status == 0
    // Additional safety: we could only allow eac_status == 0 when we have human steering allowed
    if (controls_allowed && (eac_status != 0) && (eac_status != 1) && (eac_status != 2)) {
      controls_allowed = 0;
      //puts("EPAS error! \n");
    }
  }
  //get latest steering wheel angle
  if (addr == 0x00E) {
    float angle_meas_now = (int)(((((GET_BYTE(to_push, 0) & 0x3F) << 8) + GET_BYTE(to_push, 1)) * 0.1) - 819.2);
    uint32_t ts = TIM2->CNT;
    uint32_t ts_elapsed = get_ts_elapsed(ts, tesla_ts_angle_last);

    // *** angle real time check
    // add 1 to not false trigger the violation and multiply by 25 since the check is done every 250 ms and steer angle is updated at     100Hz
    float rt_delta_angle_up = (interpolate(TESLA_LOOKUP_ANGLE_RATE_UP, tesla_speed) * 25.) + 1.;
    float rt_delta_angle_down = (interpolate(TESLA_LOOKUP_ANGLE_RATE_DOWN, tesla_speed) * 25.) + 1.;
    float highest_rt_angle = tesla_rt_angle_last + ((tesla_rt_angle_last > 0.) ? rt_delta_angle_up : rt_delta_angle_down);
    float lowest_rt_angle = tesla_rt_angle_last - ((tesla_rt_angle_last > 0.) ? rt_delta_angle_down : rt_delta_angle_up);

    if ((ts_elapsed > TESLA_RT_INTERVAL) || (controls_allowed && !tesla_controls_allowed_last)) {
      tesla_rt_angle_last = angle_meas_now;
      tesla_ts_angle_last = ts;
    }

    // check for violation;
    if (fmax_limit_check(angle_meas_now, highest_rt_angle, lowest_rt_angle)) {
      // We should not be able to STEER under these conditions
      // Other sending is fine (to allow human override)
      controls_allowed = 0;
      //puts("WARN: RT Angle - No steer allowed! \n");
    } else {
      controls_allowed = 1;
    }

    tesla_controls_allowed_last = controls_allowed;
  }
  return 1;
}

// all commands: gas/regen, friction brake and steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int tesla_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);

  // do not transmit CAN message if steering angle too high
  // DAS_steeringControl::DAS_steeringAngleRequest
  if (addr == 0x488) {
    float angle_raw = ((GET_BYTE(to_send, 0) & 0x7F) << 8) + GET_BYTE(to_send, 1);
    float desired_angle = (angle_raw * 0.1) - 1638.35;
    bool violation = 0;
    int st_enabled = GET_BYTE(to_send, 2) & 0x40;

    if (st_enabled == 0) {
      //steering is not enabled, do not check angles and do send
      tesla_desired_angle_last = desired_angle;
    } else if (controls_allowed) {
      // add 1 to not false trigger the violation
      float delta_angle_up = interpolate(TESLA_LOOKUP_ANGLE_RATE_UP, tesla_speed) + 1.;
      float delta_angle_down = interpolate(TESLA_LOOKUP_ANGLE_RATE_DOWN, tesla_speed) + 1.;
      float highest_desired_angle = tesla_desired_angle_last + ((tesla_desired_angle_last > 0.) ? delta_angle_up : delta_angle_down);
      float lowest_desired_angle = tesla_desired_angle_last - ((tesla_desired_angle_last > 0.) ? delta_angle_down : delta_angle_up);
      float TESLA_MAX_ANGLE = interpolate(TESLA_LOOKUP_MAX_ANGLE, tesla_speed) + 1.;

      //check for max angles
      violation |= fmax_limit_check(desired_angle, TESLA_MAX_ANGLE, -TESLA_MAX_ANGLE);

      //check for angle delta changes
      violation |= fmax_limit_check(desired_angle, highest_desired_angle, lowest_desired_angle);

      if (violation) {
        controls_allowed = 0;
        tx = 0;
      }
      tesla_desired_angle_last = desired_angle;
    } else {
      tx = 0;
    }
  }
  return tx;
}

static void tesla_init(int16_t param) {
  UNUSED(param);
  controls_allowed = 0;
  gmlan_switch_init(1); //init the gmlan switch with 1s timeout enabled
}

static int tesla_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {

  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);

  if (bus_num == 0) {
    // change inhibit of GTW_epasControl

    if (addr != 0x214) {
      // remove EPB_epasControl
      bus_fwd = 2; // Custom EPAS bus
    }
    if (addr == 0x101) {
      to_fwd->RDLR = GET_BYTES_04(to_fwd) | 0x4000; // 0x4000: WITH_ANGLE, 0xC000: WITH_BOTH (angle and torque)
      uint32_t checksum = (GET_BYTE(to_fwd, 1) + GET_BYTE(to_fwd, 0) + 2) & 0xFF;
      to_fwd->RDLR = GET_BYTES_04(to_fwd) & 0xFFFF;
      to_fwd->RDLR = GET_BYTES_04(to_fwd) + (checksum << 16);
    }
  }
  if (bus_num == 2) {
    // remove GTW_epasControl in forwards
    if (addr != 0x101) {
      bus_fwd = 0;  // Chassis CAN
    }
  }
  return bus_fwd;
}

const safety_hooks tesla_hooks = {
  .init = tesla_init,
  .rx = tesla_rx_hook,
  .tx = tesla_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = tesla_fwd_hook,
};
