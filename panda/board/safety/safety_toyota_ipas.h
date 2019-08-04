// uses tons from safety_toyota
// TODO: refactor to repeat less code

// IPAS override
const int32_t TOYOTA_IPAS_OVERRIDE_THRESHOLD = 200;  // disallow controls when user torque exceeds this value

// 2m/s are added to be less restrictive
const struct lookup_t LOOKUP_ANGLE_RATE_UP = {
  {2., 7., 17.},
  {5., .8, .15}};

const struct lookup_t LOOKUP_ANGLE_RATE_DOWN = {
  {2., 7., 17.},
  {5., 3.5, .4}};

const float RT_ANGLE_FUDGE = 1.5;     // for RT checks allow 50% more angle change
const float CAN_TO_DEG = 2. / 3.;      // convert angles from CAN unit to degrees

int ipas_state = 1;                    // 1 disabled, 3 executing angle control, 5 override
int angle_control = 0;                 // 1 if direct angle control packets are seen
float speed = 0.;

struct sample_t angle_meas;            // last 3 steer angles
struct sample_t torque_driver;         // last 3 driver steering torque

// state of angle limits
int16_t desired_angle_last = 0;        // last desired steer angle
int16_t rt_angle_last = 0;             // last desired torque for real time check
uint32_t ts_angle_last = 0;

int controls_allowed_last = 0;


static void toyota_ipas_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  // check standard toyota stuff as well
  toyota_rx_hook(to_push);

  int addr = GET_ADDR(to_push);

  if (addr == 0x260) {
    // get driver steering torque
    int16_t torque_driver_new = (GET_BYTE(to_push, 1) << 8) | GET_BYTE(to_push, 2);

    // update array of samples
    update_sample(&torque_driver, torque_driver_new);
  }

  // get steer angle
  if (addr == 0x25) {
    int angle_meas_new = ((GET_BYTE(to_push, 0) & 0xF) << 8) | GET_BYTE(to_push, 1);
    uint32_t ts = TIM2->CNT;

    angle_meas_new = to_signed(angle_meas_new, 12);

    // update array of samples
    update_sample(&angle_meas, angle_meas_new);

    // *** angle real time check
    // add 1 to not false trigger the violation and multiply by 20 since the check is done every 250ms and steer angle is updated at 80Hz
    int rt_delta_angle_up = ((int)(RT_ANGLE_FUDGE * ((interpolate(LOOKUP_ANGLE_RATE_UP, speed) * 20. * CAN_TO_DEG) + 1.)));
    int rt_delta_angle_down = ((int)(RT_ANGLE_FUDGE * ((interpolate(LOOKUP_ANGLE_RATE_DOWN, speed) * 20. * CAN_TO_DEG) + 1.)));
    int highest_rt_angle = rt_angle_last + ((rt_angle_last > 0) ? rt_delta_angle_up : rt_delta_angle_down);
    int lowest_rt_angle = rt_angle_last - ((rt_angle_last > 0) ? rt_delta_angle_down : rt_delta_angle_up);

    // every RT_INTERVAL or when controls are turned on, set the new limits
    uint32_t ts_elapsed = get_ts_elapsed(ts, ts_angle_last);
    if ((ts_elapsed > TOYOTA_RT_INTERVAL) || (controls_allowed && !controls_allowed_last)) {
      rt_angle_last = angle_meas_new;
      ts_angle_last = ts;
    }

    // check for violation
    if (angle_control &&
        ((angle_meas_new < lowest_rt_angle) ||
         (angle_meas_new > highest_rt_angle))) {
      controls_allowed = 0;
    }

    controls_allowed_last = controls_allowed;
  }

  // get speed
  if (addr == 0xb4) {
    speed = ((float)((GET_BYTE(to_push, 5) << 8) | GET_BYTE(to_push, 6))) * 0.01 / 3.6;
  }

  // get ipas state
  if (addr == 0x262) {
    ipas_state = GET_BYTE(to_push, 0) & 0xf;
  }

  // exit controls on high steering override
  if (angle_control && ((torque_driver.min > TOYOTA_IPAS_OVERRIDE_THRESHOLD) ||
                        (torque_driver.max < -TOYOTA_IPAS_OVERRIDE_THRESHOLD) ||
                        (ipas_state==5))) {
    controls_allowed = 0;
  }
}

static int toyota_ipas_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int bypass_standard_tx_hook = 0;
  int bus = GET_BUS(to_send);
  int addr = GET_ADDR(to_send);

  // Check if msg is sent on BUS 0
  if (bus == 0) {

    // STEER ANGLE
    if ((addr == 0x266) || (addr == 0x167)) {

      angle_control = 1;   // we are in angle control mode
      int desired_angle = ((GET_BYTE(to_send, 0) & 0xF) << 8) | GET_BYTE(to_send, 1);
      int ipas_state_cmd = GET_BYTE(to_send, 0) >> 4;
      bool violation = 0;

      desired_angle = to_signed(desired_angle, 12);

      if (controls_allowed) {
        // add 1 to not false trigger the violation
        float delta_angle_float;
        delta_angle_float = (interpolate(LOOKUP_ANGLE_RATE_UP, speed) * CAN_TO_DEG) + 1.;
        int delta_angle_up = (int) (delta_angle_float);
        delta_angle_float = (interpolate(LOOKUP_ANGLE_RATE_DOWN, speed) * CAN_TO_DEG) + 1.;
        int delta_angle_down = (int) (delta_angle_float);

        int highest_desired_angle = desired_angle_last + ((desired_angle_last > 0) ? delta_angle_up : delta_angle_down);
        int lowest_desired_angle = desired_angle_last - ((desired_angle_last > 0) ? delta_angle_down : delta_angle_up);
        if ((desired_angle > highest_desired_angle) ||
            (desired_angle < lowest_desired_angle)){
          violation = 1;
          controls_allowed = 0;
        }
      }

      // desired steer angle should be the same as steer angle measured when controls are off
      if ((!controls_allowed) &&
           ((desired_angle < (angle_meas.min - 1)) ||
            (desired_angle > (angle_meas.max + 1)) ||
            (ipas_state_cmd != 1))) {
        violation = 1;
      }

      desired_angle_last = desired_angle;

      if (violation) {
        tx = 0;
      }
      bypass_standard_tx_hook = 1;
    }
  }

  // check standard toyota stuff as well if addr isn't IPAS related
  if (!bypass_standard_tx_hook) {
    tx &= toyota_tx_hook(to_send);
  }

  return tx;
}

const safety_hooks toyota_ipas_hooks = {
  .init = toyota_init,
  .rx = toyota_ipas_rx_hook,
  .tx = toyota_ipas_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = toyota_fwd_hook,
};
