// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      regen paddle
//      accel rising edge
//      brake rising edge
//      brake > 0mph


// 2m/s are added to be less restrictive
const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_UP = {
  {2., 7., 17.},
  {5., .8, .25}};

const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_DOWN = {
  {2., 7., 17.},
  {5., 3.5, .8}};

const struct lookup_t TESLA_LOOKUP_MAX_ANGLE = {
  {2., 29., 38.},
  {410.,92.,36.}};

 const int TESLA_RT_INTERVAL = 250000;    // 250ms between real time checks

struct sample_t tesla_angle_meas;            // last 3 steer angles

// state of angle limits
int tesla_desired_angle_last = 0;        // last desired steer angle
int16_t  tesla_rt_angle_last = 0.;             // last real time angle
uint32_t tesla_ts_angle_last = 0;


int tesla_controls_allowed_last = 0;


int tesla_brake_prev = 0;
int tesla_gas_prev = 0;
int tesla_speed = 0;
int current_car_time = -1;
int time_at_last_stalk_pull = -1;
int eac_status = 0;

int tesla_ignition_started = 0;

// interp function that holds extreme values
float tesla_interpolate(struct lookup_t xy, float x) {
  int size = sizeof(xy.x) / sizeof(xy.x[0]);
  // x is lower than the first point in the x array. Return the first point
  if (x <= xy.x[0]) {
    return xy.y[0];

  } else {
    // find the index such that (xy.x[i] <= x < xy.x[i+1]) and linearly interp
    for (int i=0; i < size-1; i++) {
      if (x < xy.x[i+1]) {
        float x0 = xy.x[i];
        float y0 = xy.y[i];
        float dx = xy.x[i+1] - x0;
        float dy = xy.y[i+1] - y0;
        // dx should not be zero as xy.x is supposed ot be monotonic
        if (dx <= 0.) dx = 0.0001;
        return dy * (x - x0) / dx + y0;
      }
    }
    // if no such point is found, then x > xy.x[size-1]. Return last point
    return xy.y[size - 1];
  }
}


static void tesla_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  set_gmlan_digital_output(GMLAN_HIGH);
  reset_gmlan_switch_timeout(); //we're still in tesla safety mode, reset the timeout counter and make sure our output is enabled
  
  //int bus_number = (to_push->RDTR >> 4) & 0xFF;
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

   // Record the current car time in current_car_time (for use with double-pulling cruise stalk)
  if (addr == 0x318) {
    int hour = (to_push->RDLR & 0x1F000000) >> 24;
    int minute = (to_push->RDHR & 0x3F00) >> 8;
    int second = (to_push->RDLR & 0x3F0000) >> 16;
    current_car_time = (hour * 3600) + (minute * 60) + second;
  }
  
  if (addr == 0x45) {
    // 6 bits starting at position 0
    int lever_position = (to_push->RDLR & 0x3F);
    if (lever_position == 2) { // pull forward
      // activate openpilot
      // TODO: uncomment the if to use double pull to activate
      //if (current_car_time <= time_at_last_stalk_pull + 1 && current_car_time != -1 && time_at_last_stalk_pull != -1) {
          controls_allowed = 1;
      //}
      time_at_last_stalk_pull = current_car_time;
    } else if (lever_position == 1) { // push towards the back
      // deactivate openpilot
      controls_allowed = 0;
    }
  }  
  
  // Detect drive rail on (ignition) (start recording)
  if (addr == 0x348) {
    // GTW_status
    int drive_rail_on = (to_push->RDLR & 0x0001);
    tesla_ignition_started = drive_rail_on == 1;
  }

  // exit controls on brake press
  // DI_torque2::DI_brakePedal 0x118
  if (addr == 0x118) {
    // 1 bit at position 16
    if (((to_push->RDLR & 0x8000)) >> 15 == 1) {
      //disable break cancel by commenting line below
      //controls_allowed = 0;
    }
    //get vehicle speed in m/2. Tesla gives MPH
    tesla_speed = ((((((to_push->RDLR >> 24) & 0x0F) << 8) + (( to_push->RDLR >> 16) & 0xFF)) * 0.05 -25)*1.609/3.6);
    if (tesla_speed < 0) {
      tesla_speed = 0;
    }
  }  
  
  // exit controls on EPAS error
  // EPAS_sysStatus::EPAS_eacStatus 0x370
  if (addr == 0x370) {
    // if EPAS_eacStatus is not 1 or 2, disable control
    eac_status = ((to_push->RDHR >> 21)) & 0x7;
    // For human steering override we must not disable controls when eac_status == 0
    // Additional safety: we could only allow eac_status == 0 when we have human steerign allowed
    if ((controls_allowed == 1) && (eac_status != 0) && (eac_status != 1) && (eac_status != 2)) {
      controls_allowed = 0;
      puts("EPAS error! \n");
    }
  }  
  //get latest steering wheel angle
  if (addr == 0x00E) {
    int angle_meas_now = (int) ((((to_push->RDLR & 0x3F) <<8) + ((to_push->RDLR >> 8) & 0xFF)) * 0.1 - 819.2);
    uint32_t ts = TIM2->CNT;
    uint32_t ts_elapsed = get_ts_elapsed(ts, tesla_ts_angle_last);

    // *** angle real time check
    // add 1 to not false trigger the violation and multiply by 25 since the check is done every 250 ms and steer angle is updated at     100Hz
    int rt_delta_angle_up = ((int)((tesla_interpolate(TESLA_LOOKUP_ANGLE_RATE_UP, tesla_speed) * 25. + 1.)));
    int rt_delta_angle_down = ((int)( (tesla_interpolate(TESLA_LOOKUP_ANGLE_RATE_DOWN, tesla_speed) * 25. + 1.)));
    int highest_rt_angle = tesla_rt_angle_last + (tesla_rt_angle_last > 0 ? rt_delta_angle_up:rt_delta_angle_down);
    int lowest_rt_angle = tesla_rt_angle_last - (tesla_rt_angle_last > 0 ? rt_delta_angle_down:rt_delta_angle_up);

    if ((ts_elapsed > TESLA_RT_INTERVAL) || (controls_allowed && !tesla_controls_allowed_last)) {
	    tesla_rt_angle_last = angle_meas_now;
      tesla_ts_angle_last = ts;
    }

    // update array of samples
    update_sample(&tesla_angle_meas, angle_meas_now);

    
    // check for violation;
    if (max_limit_check(angle_meas_now, highest_rt_angle, lowest_rt_angle)) {
      controls_allowed = 0;
      puts("RT Angle Error! \n");
    }

    tesla_controls_allowed_last = controls_allowed;   
 
  }
}

// all commands: gas/regen, friction brake and steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int tesla_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  uint32_t addr;
  int angle_raw;
  int desired_angle;

  addr = to_send->RIR >> 21;
  
  // do not transmit CAN message if steering angle too high
  // DAS_steeringControl::DAS_steeringAngleRequest
  if (addr == 0x488) {
    angle_raw = ((to_send->RDLR & 0x7F) << 8) + ((to_send->RDLR & 0xFF00) >> 8);
    desired_angle = angle_raw * 0.1 - 1638.35;
    int16_t violation = 0;

    if (controls_allowed) {

			 // add 1 to not false trigger the violation
       int delta_angle_up = (int) (tesla_interpolate(TESLA_LOOKUP_ANGLE_RATE_UP, tesla_speed)* 25.  + 1.);
       int delta_angle_down = (int) (tesla_interpolate(TESLA_LOOKUP_ANGLE_RATE_DOWN, tesla_speed)* 25. + 1.);
       int highest_desired_angle = tesla_desired_angle_last + (tesla_desired_angle_last > 0 ? delta_angle_up:delta_angle_down);
       int lowest_desired_angle = tesla_desired_angle_last - (tesla_desired_angle_last > 0 ? delta_angle_down:delta_angle_up);
       int TESLA_MAX_ANGLE = (int) (tesla_interpolate(TESLA_LOOKUP_MAX_ANGLE, tesla_speed) +1.);

       if (max_limit_check(desired_angle, highest_desired_angle, lowest_desired_angle)){
         violation = 1;
         controls_allowed = 0;
         puts("Angle limit - delta! \n");
       }
       if (max_limit_check(desired_angle, TESLA_MAX_ANGLE, -TESLA_MAX_ANGLE)) {
			     violation =1;
           controls_allowed = 0;
           puts("Angle limit - max! \n");
       }
	  }

// makes no sense to have angle limits when not engaged
//    if ((!controls_allowed) && max_limit_check(desired_angle, tesla_angle_meas.max + 1, tesla_angle_meas.min -1)) {
//       violation = 1;
//       puts("Angle limit when not engaged! \n");
//    }

    tesla_desired_angle_last = desired_angle;

    if (violation) {
      return false;
    }
    return true;  
  }  
  return true;
}

static int tesla_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  // LIN is not used on the Tesla
  return false;
}

static void tesla_init(int16_t param) {
  controls_allowed = 0;
  tesla_ignition_started = 0;
  gmlan_switch_init(1); //init the gmlan switch with 1s timeout enabled
}

static int tesla_ign_hook() {
  return tesla_ignition_started;
}

static int tesla_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  
  int32_t addr = to_fwd->RIR >> 21;
  
  if (bus_num == 0) {
    
    // change inhibit of GTW_epasControl
    if (addr == 0x101) {
      to_fwd->RDLR = to_fwd->RDLR | 0x4000; // 0x4000: WITH_ANGLE, 0xC000: WITH_BOTH (angle and torque)
      int checksum = (((to_fwd->RDLR & 0xFF00) >> 8) + (to_fwd->RDLR & 0xFF) + 2) & 0xFF;
      to_fwd->RDLR = to_fwd->RDLR & 0xFFFF;
      to_fwd->RDLR = to_fwd->RDLR + (checksum << 16);
      return 2;
    }

    // remove EPB_epasControl
    if (addr == 0x214) {
      return false;
    }

    return 2; // Custom EPAS bus
  }
  if (bus_num == 2) {
    
    // remove GTW_epasControl in forwards
   if (addr == 0x101) {
     return false;
   }

    return 0; // Chassis CAN
  }
  return false;
}

const safety_hooks tesla_hooks = {
  .init = tesla_init,
  .rx = tesla_rx_hook,
  .tx = tesla_tx_hook,
  .tx_lin = tesla_tx_lin_hook,
  .ignition = tesla_ign_hook,
  .fwd = tesla_fwd_hook,
};
