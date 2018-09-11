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

  if ((to_push->RIR>>21) == 0x260) {
    // get driver steering torque
    int16_t torque_driver_new = (((to_push->RDLR) & 0xFF00) | ((to_push->RDLR >> 16) & 0xFF));

    // update array of samples
    update_sample(&torque_driver, torque_driver_new);
  }

  // get steer angle
  if ((to_push->RIR>>21) == 0x25) {
    int angle_meas_new = ((to_push->RDLR & 0xf) << 8) + ((to_push->RDLR & 0xff00) >> 8);
    uint32_t ts = TIM2->CNT;

    angle_meas_new = to_signed(angle_meas_new, 12);

    // update array of samples
    update_sample(&angle_meas, angle_meas_new);

    // *** angle real time check
    // add 1 to not false trigger the violation and multiply by 20 since the check is done every 250ms and steer angle is updated at 80Hz
    int rt_delta_angle_up = ((int)(RT_ANGLE_FUDGE * (interpolate(LOOKUP_ANGLE_RATE_UP, speed) * 20. * CAN_TO_DEG + 1.)));
    int rt_delta_angle_down = ((int)(RT_ANGLE_FUDGE * (interpolate(LOOKUP_ANGLE_RATE_DOWN, speed) * 20. * CAN_TO_DEG + 1.)));
    int highest_rt_angle = rt_angle_last + (rt_angle_last > 0? rt_delta_angle_up:rt_delta_angle_down);
    int lowest_rt_angle = rt_angle_last - (rt_angle_last > 0? rt_delta_angle_down:rt_delta_angle_up);

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
  if ((to_push->RIR>>21) == 0xb4) {
    speed = ((float) (((to_push->RDHR) & 0xFF00) | ((to_push->RDHR >> 16) & 0xFF))) * 0.01 / 3.6;
  }

  // get ipas state
  if ((to_push->RIR>>21) == 0x262) {
    ipas_state = (to_push->RDLR & 0xf);
  }

  // exit controls on high steering override
  if (angle_control && ((torque_driver.min > TOYOTA_IPAS_OVERRIDE_THRESHOLD) ||
                        (torque_driver.max < -TOYOTA_IPAS_OVERRIDE_THRESHOLD) ||
                        (ipas_state==5))) {
    controls_allowed = 0;
  }
}

static int toyota_ipas_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  // Check if msg is sent on BUS 0
  if (((to_send->RDTR >> 4) & 0xF) == 0) {

    // STEER ANGLE
    if (((to_send->RIR>>21) == 0x266) || ((to_send->RIR>>21) == 0x167)) {

      angle_control = 1;   // we are in angle control mode
      int desired_angle = ((to_send->RDLR & 0xf) << 8) + ((to_send->RDLR & 0xff00) >> 8);
      int ipas_state_cmd = ((to_send->RDLR & 0xff) >> 4);
      int16_t violation = 0;

      desired_angle = to_signed(desired_angle, 12);

      if (controls_allowed) {
        // add 1 to not false trigger the violation
        int delta_angle_up = (int) (interpolate(LOOKUP_ANGLE_RATE_UP, speed) * CAN_TO_DEG + 1.);
        int delta_angle_down = (int) (interpolate(LOOKUP_ANGLE_RATE_DOWN, speed) * CAN_TO_DEG + 1.);
        int highest_desired_angle = desired_angle_last + (desired_angle_last > 0? delta_angle_up:delta_angle_down);
        int lowest_desired_angle = desired_angle_last - (desired_angle_last > 0? delta_angle_down:delta_angle_up);
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
        return false;
      }

      return true;
    }
  }

  // check standard toyota stuff as well
  return toyota_tx_hook(to_send);
}

const safety_hooks toyota_ipas_hooks = {
  .init = toyota_init,
  .rx = toyota_ipas_rx_hook,
  .tx = toyota_ipas_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = toyota_fwd_hook,
};

