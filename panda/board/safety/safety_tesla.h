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
    {15., 15.8, 15.25}};

const struct lookup_t TESLA_LOOKUP_ANGLE_RATE_DOWN = {
    {2., 7., 17.},
    {15., 13.5, 15.8}};

const struct lookup_t TESLA_LOOKUP_MAX_ANGLE = {
    {2., 29., 38.},
    {410., 492., 436.}};

const int TESLA_RT_INTERVAL = 250000; // 250ms between real time checks

struct sample_t tesla_angle_meas; // last 3 steer angles

// state of angle limits
int tesla_desired_angle_last = 0; // last desired steer angle
int16_t tesla_rt_angle_last = 0.; // last real time angle
uint32_t tesla_ts_angle_last = 0;

int tesla_controls_allowed_last = 0;
int steer_allowed = 1;

int tesla_brake_prev = 0;
int tesla_gas_prev = 0;
int tesla_speed = 0;
int current_car_time = -1;
int time_at_last_stalk_pull = -1;
int eac_status = 0;

int tesla_ignition_started = 0;

static int add_tesla_crc(CAN_FIFOMailBox_TypeDef *msg , int msg_len) {
  //"""Calculate CRC8 using 1D poly, FF start, FF end"""
  int crc_lookup[256] = { 0x00, 0x1D, 0x3A, 0x27, 0x74, 0x69, 0x4E, 0x53, 0xE8, 0xF5, 0xD2, 0xCF, 0x9C, 0x81, 0xA6, 0xBB, 
    0xCD, 0xD0, 0xF7, 0xEA, 0xB9, 0xA4, 0x83, 0x9E, 0x25, 0x38, 0x1F, 0x02, 0x51, 0x4C, 0x6B, 0x76, 
    0x87, 0x9A, 0xBD, 0xA0, 0xF3, 0xEE, 0xC9, 0xD4, 0x6F, 0x72, 0x55, 0x48, 0x1B, 0x06, 0x21, 0x3C, 
    0x4A, 0x57, 0x70, 0x6D, 0x3E, 0x23, 0x04, 0x19, 0xA2, 0xBF, 0x98, 0x85, 0xD6, 0xCB, 0xEC, 0xF1, 
    0x13, 0x0E, 0x29, 0x34, 0x67, 0x7A, 0x5D, 0x40, 0xFB, 0xE6, 0xC1, 0xDC, 0x8F, 0x92, 0xB5, 0xA8, 
    0xDE, 0xC3, 0xE4, 0xF9, 0xAA, 0xB7, 0x90, 0x8D, 0x36, 0x2B, 0x0C, 0x11, 0x42, 0x5F, 0x78, 0x65, 
    0x94, 0x89, 0xAE, 0xB3, 0xE0, 0xFD, 0xDA, 0xC7, 0x7C, 0x61, 0x46, 0x5B, 0x08, 0x15, 0x32, 0x2F, 
    0x59, 0x44, 0x63, 0x7E, 0x2D, 0x30, 0x17, 0x0A, 0xB1, 0xAC, 0x8B, 0x96, 0xC5, 0xD8, 0xFF, 0xE2, 
    0x26, 0x3B, 0x1C, 0x01, 0x52, 0x4F, 0x68, 0x75, 0xCE, 0xD3, 0xF4, 0xE9, 0xBA, 0xA7, 0x80, 0x9D, 
    0xEB, 0xF6, 0xD1, 0xCC, 0x9F, 0x82, 0xA5, 0xB8, 0x03, 0x1E, 0x39, 0x24, 0x77, 0x6A, 0x4D, 0x50, 
    0xA1, 0xBC, 0x9B, 0x86, 0xD5, 0xC8, 0xEF, 0xF2, 0x49, 0x54, 0x73, 0x6E, 0x3D, 0x20, 0x07, 0x1A, 
    0x6C, 0x71, 0x56, 0x4B, 0x18, 0x05, 0x22, 0x3F, 0x84, 0x99, 0xBE, 0xA3, 0xF0, 0xED, 0xCA, 0xD7, 
    0x35, 0x28, 0x0F, 0x12, 0x41, 0x5C, 0x7B, 0x66, 0xDD, 0xC0, 0xE7, 0xFA, 0xA9, 0xB4, 0x93, 0x8E, 
    0xF8, 0xE5, 0xC2, 0xDF, 0x8C, 0x91, 0xB6, 0xAB, 0x10, 0x0D, 0x2A, 0x37, 0x64, 0x79, 0x5E, 0x43, 
    0xB2, 0xAF, 0x88, 0x95, 0xC6, 0xDB, 0xFC, 0xE1, 0x5A, 0x47, 0x60, 0x7D, 0x2E, 0x33, 0x14, 0x09, 
    0x7F, 0x62, 0x45, 0x58, 0x0B, 0x16, 0x31, 0x2C, 0x97, 0x8A, 0xAD, 0xB0, 0xE3, 0xFE, 0xD9, 0xC4 };
  int crc = 0xFF;
  for (int x = 0; x < msg_len; x++) {
    int v = 0;
    if (x <= 3) {
      v = (msg->RDLR >> (x * 8)) & 0xFF;
    } else {
      v = (msg->RDHR >> ( (x-4) * 8)) & 0xFF;
    }
    crc = crc_lookup[crc ^ v];
  }
  crc = crc ^ 0xFF;
  return crc;
}

static int add_tesla_cksm(CAN_FIFOMailBox_TypeDef *msg , int msg_id, int msg_len) {
  int cksm = (0xFF & msg_id) + (0xFF & (msg_id >> 8));
  for (int x = 0; x < msg_len; x++) {
    int v = 0;
    if (x <= 3) {
      v = (msg->RDLR >> (x * 8)) & 0xFF;
    } else {
      v = (msg->RDHR >> ( (x-4) * 8)) & 0xFF;
    }
    cksm = (cksm + v) & 0xFF;
  }
  return cksm;
}

// interp function that holds extreme values
float tesla_interpolate(struct lookup_t xy, float x)
{
  int size = sizeof(xy.x) / sizeof(xy.x[0]);
  // x is lower than the first point in the x array. Return the first point
  if (x <= xy.x[0])
  {
    return xy.y[0];
  }
  else
  {
    // find the index such that (xy.x[i] <= x < xy.x[i+1]) and linearly interp
    for (int i = 0; i < size - 1; i++)
    {
      if (x < xy.x[i + 1])
      {
        float x0 = xy.x[i];
        float y0 = xy.y[i];
        float dx = xy.x[i + 1] - x0;
        float dy = xy.y[i + 1] - y0;
        // dx should not be zero as xy.x is supposed ot be monotonic
        if (dx <= 0.)
          dx = 0.0001;
        return dy * (x - x0) / dx + y0;
      }
    }
    // if no such point is found, then x > xy.x[size-1]. Return last point
    return xy.y[size - 1];
  }
}

static void tesla_rx_hook(CAN_FIFOMailBox_TypeDef *to_push)
{
  set_gmlan_digital_output(GMLAN_HIGH);
  reset_gmlan_switch_timeout(); //we're still in tesla safety mode, reset the timeout counter and make sure our output is enabled

  //int bus_number = (to_push->RDTR >> 4) & 0xFF;
  uint32_t addr;
  if (to_push->RIR & 4)
  {
    // Extended
    // Not looked at, but have to be separated
    // to avoid address collision
    addr = to_push->RIR >> 3;
  }
  else
  {
    // Normal
    addr = to_push->RIR >> 21;
  }

  // Record the current car time in current_car_time (for use with double-pulling cruise stalk)
  if (addr == 0x318)
  {
    int hour = (to_push->RDLR & 0x1F000000) >> 24;
    int minute = (to_push->RDHR & 0x3F00) >> 8;
    int second = (to_push->RDLR & 0x3F0000) >> 16;
    current_car_time = (hour * 3600) + (minute * 60) + second;
  }

  if (addr == 0x45)
  {
    // 6 bits starting at position 0
    int lever_position = (to_push->RDLR & 0x3F);
    if (lever_position == 2)
    { // pull forward
      // activate openpilot
      // TODO: uncomment the if to use double pull to activate
      //if (current_car_time <= time_at_last_stalk_pull + 1 && current_car_time != -1 && time_at_last_stalk_pull != -1) {
      controls_allowed = 1;
      //}
      time_at_last_stalk_pull = current_car_time;
    }
    else if (lever_position == 1)
    { // push towards the back
      // deactivate openpilot
      controls_allowed = 0;
    }
  }

  // Detect drive rail on (ignition) (start recording)
  if (addr == 0x348)
  {
    // GTW_status
    int drive_rail_on = (to_push->RDLR & 0x0001);
    tesla_ignition_started = drive_rail_on == 1;
  }

  // exit controls on brake press
  // DI_torque2::DI_brakePedal 0x118
  if (addr == 0x118)
  {
    // 1 bit at position 16
    if (((to_push->RDLR & 0x8000)) >> 15 == 1)
    {
      //disable break cancel by commenting line below
      //controls_allowed = 0;
    }
    //get vehicle speed in m/2. Tesla gives MPH
    tesla_speed = ((((((to_push->RDLR >> 24) & 0x0F) << 8) + ((to_push->RDLR >> 16) & 0xFF)) * 0.05 - 25) * 1.609 / 3.6);
    if (tesla_speed < 0)
    {
      tesla_speed = 0;
    }
  }

  // exit controls on EPAS error
  // EPAS_sysStatus::EPAS_eacStatus 0x370
  if (addr == 0x370)
  {
    // if EPAS_eacStatus is not 1 or 2, disable control
    eac_status = ((to_push->RDHR >> 21)) & 0x7;
    // For human steering override we must not disable controls when eac_status == 0
    // Additional safety: we could only allow eac_status == 0 when we have human steerign allowed
    if ((controls_allowed == 1) && (eac_status != 0) && (eac_status != 1) && (eac_status != 2))
    {
      controls_allowed = 0;
      puts("EPAS error! \n");
    }
  }
  //get latest steering wheel angle
  if (addr == 0x00E)
  {
    int angle_meas_now = (int)((((to_push->RDLR & 0x3F) << 8) + ((to_push->RDLR >> 8) & 0xFF)) * 0.1 - 819.2);
    uint32_t ts = TIM2->CNT;
    uint32_t ts_elapsed = get_ts_elapsed(ts, tesla_ts_angle_last);

    // *** angle real time check
    // add 1 to not false trigger the violation and multiply by 25 since the check is done every 250 ms and steer angle is updated at     100Hz
    int rt_delta_angle_up = ((int)((tesla_interpolate(TESLA_LOOKUP_ANGLE_RATE_UP, tesla_speed) * 25. + 1.)));
    int rt_delta_angle_down = ((int)((tesla_interpolate(TESLA_LOOKUP_ANGLE_RATE_DOWN, tesla_speed) * 25. + 1.)));
    int highest_rt_angle = tesla_rt_angle_last + (tesla_rt_angle_last > 0 ? rt_delta_angle_up : rt_delta_angle_down);
    int lowest_rt_angle = tesla_rt_angle_last - (tesla_rt_angle_last > 0 ? rt_delta_angle_down : rt_delta_angle_up);

    if ((ts_elapsed > TESLA_RT_INTERVAL) || (controls_allowed && !tesla_controls_allowed_last))
    {
      tesla_rt_angle_last = angle_meas_now;
      tesla_ts_angle_last = ts;
    }

    // update array of samples
    update_sample(&tesla_angle_meas, angle_meas_now);

    // check for violation;
    if (max_limit_check(angle_meas_now, highest_rt_angle, lowest_rt_angle))
    {
      // We should not be able to STEER under these conditions
      // Other sending is fine (to allow human override)
      steer_allowed = 0;
      puts("WARN: RT Angle - No steer allowed! \n");
    }
    else
    {
      steer_allowed = 1;
    }

    tesla_controls_allowed_last = controls_allowed;
  }
}

// all commands: gas/regen, friction brake and steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int tesla_tx_hook(CAN_FIFOMailBox_TypeDef *to_send)
{

  uint32_t addr;
  int angle_raw;
  int desired_angle;

  addr = to_send->RIR >> 21;

  // do not transmit CAN message if steering angle too high
  // DAS_steeringControl::DAS_steeringAngleRequest
  if (addr == 0x488)
  {
    angle_raw = ((to_send->RDLR & 0x7F) << 8) + ((to_send->RDLR & 0xFF00) >> 8);
    desired_angle = angle_raw * 0.1 - 1638.35;
    int16_t violation = 0;
    int st_enabled = (to_send->RDLR & 0x400000) >> 22;

    if (st_enabled == 0) {
      //steering is not enabled, do not check angles and do send
      tesla_desired_angle_last = desired_angle;
      return true;
    }

    if (controls_allowed)
    {
      if (steer_allowed)
      {

        // add 1 to not false trigger the violation
        int delta_angle_up = (int)(tesla_interpolate(TESLA_LOOKUP_ANGLE_RATE_UP, tesla_speed) * 25. + 1.);
        int delta_angle_down = (int)(tesla_interpolate(TESLA_LOOKUP_ANGLE_RATE_DOWN, tesla_speed) * 25. + 1.);
        int highest_desired_angle = tesla_desired_angle_last + (tesla_desired_angle_last > 0 ? delta_angle_up : delta_angle_down);
        int lowest_desired_angle = tesla_desired_angle_last - (tesla_desired_angle_last > 0 ? delta_angle_down : delta_angle_up);
        int TESLA_MAX_ANGLE = (int)(tesla_interpolate(TESLA_LOOKUP_MAX_ANGLE, tesla_speed) + 1.);

        if (max_limit_check(desired_angle, highest_desired_angle, lowest_desired_angle))
        {
          violation = 1;
          controls_allowed = 0;
          puts("Angle limit - delta! \n");
        }
        if (max_limit_check(desired_angle, TESLA_MAX_ANGLE, -TESLA_MAX_ANGLE))
        {
          violation = 1;
          controls_allowed = 0;
          puts("Angle limit - max! \n");
        }
      }
      else
      {
        violation = 1;
        controls_allowed = 0;
        puts("Steering commads disallowed");
      }
    }

    // makes no sense to have angle limits when not engaged
    //    if ((!controls_allowed) && max_limit_check(desired_angle, tesla_angle_meas.max + 1, tesla_angle_meas.min -1)) {
    //       violation = 1;
    //       puts("Angle limit when not engaged! \n");
    //    }

    tesla_desired_angle_last = desired_angle;

    if (violation)
    {
      return false;
    }
    return true;
  }
  return true;
}

static int tesla_tx_lin_hook(int lin_num, uint8_t *data, int len)
{
  // LIN is not used on the Tesla
  return false;
}

static void tesla_init(int16_t param)
{
  controls_allowed = 0;
  tesla_ignition_started = 0;
  gmlan_switch_init(1); //init the gmlan switch with 1s timeout enabled
}

static int tesla_ign_hook()
{
  return tesla_ignition_started;
}

static void tesla_fwd_to_radar_as_is(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  CAN_FIFOMailBox_TypeDef to_send;
  to_send.RIR = to_fwd->RIR | 1; // TXRQ
  to_send.RDTR = to_fwd->RDTR;
  to_send.RDLR = to_fwd->RDLR;
  to_send.RDHR = to_fwd->RDHR;
  can_send(&to_send, bus_num);
}



static void tesla_fwd_to_radar_modded(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  int32_t addr = to_fwd->RIR >> 21;
  CAN_FIFOMailBox_TypeDef to_send;
  to_send.RIR = to_fwd->RIR | 1; // TXRQ
  to_send.RDTR = to_fwd->RDTR;
  to_send.RDLR = to_fwd->RDLR;
  to_send.RDHR = to_fwd->RDHR;
  uint32_t addr_mask = 0x001FFFFF;
  //now modd
  if (addr == 0x405 )
  {
    to_send.RIR = (0x2B9 << 21) + (addr_mask & (to_fwd->RIR | 1));
    if ((to_send.RDLR & 0x10) == 0x10)
    {
      int rec = to_send.RDLR &  0xFF;
      if (rec == 0x12) {
        to_send.RDHR = 0x36333537;
        to_send.RDLR = 0x38304600 | rec;
      }
      if (rec == 0x10) {
        to_send.RDLR = 0x00000000 | rec;
        to_send.RDHR = 0x4a593500;
      }
      if (rec == 0x11) {
        to_send.RDLR = 0x31415300 | rec;
        to_send.RDHR = 0x46373248;
      }
    }
  }
  if (addr == 0x398 )
  {
    //change frontradarHW = 1 and dashw = 1
    //SG_ GTW_dasHw : 7|2@0+ (1,0) [0|0] ""  NEO
    //SG_ GTW_parkAssistInstalled : 11|2@0+ (1,0) [0|0] ""  NEO
    to_send.RDLR = to_send.RDLR & 0xFFFFF33F;
    to_send.RDLR = to_send.RDLR | 0x440;
    to_send.RDHR = to_send.RDHR & 0xCFFFFFFF;
    to_send.RDHR = to_send.RDHR | 0x10000000;
    to_send.RIR = (0x2A9 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x00E )
  {
    to_send.RIR = (0x199 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x20A )
  {
    to_send.RIR = (0x159 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x115 )
  {
    
    int counter = ((to_fwd->RDLR & 0x000F0000) >> 16 ) & 0x0F;

    to_send.RDTR = (to_fwd->RDTR & 0xFFFFFFF0) | 0x06;
    to_send.RIR = (0x129 << 21) + (addr_mask & (to_fwd->RIR | 1));
    to_send.RDLR = 0x00000000 ;
    int cksm = (0x16 + (counter << 4)) & 0xFF;
    to_send.RDHR = (cksm << 8 ) + (counter << 4);
    can_send(&to_send, bus_num);

    to_send.RDTR = (to_fwd->RDTR & 0xFFFFFFF0) | 0x08;
    to_send.RIR = (0x149 << 21) + (addr_mask & (to_fwd->RIR | 1));
    to_send.RDLR = 0x6A022600;
    cksm = (0x95 + (counter << 4)) & 0xFF;
    to_send.RDHR = 0x000F04AA | (counter << 20) | (cksm << 24);
    can_send(&to_send, bus_num);

    

    to_send.RDTR = (to_fwd->RDTR & 0xFFFFFFF0) | 0x05;
    to_send.RIR = (0x1A9 << 21) + (addr_mask & (to_fwd->RIR | 1));
    to_send.RDLR = 0x000C0000 | (counter << 28);
    cksm = (0x49 + 0x0C + (counter << 4)) & 0xFF;
    to_send.RDHR = cksm;
  }

  if (addr == 0x118 )
  {
    to_send.RIR = (0x119 << 21) + (addr_mask & (to_fwd->RIR | 1));
    can_send(&to_send, bus_num);

    int counter = to_fwd->RDHR  & 0x0F;
    to_send.RIR = (0x169 << 21) + (addr_mask & (to_fwd->RIR | 1));
    to_send.RDTR = (to_fwd->RDTR & 0xFFFFFFF0) | 0x08;
    int32_t speed_kph = (((0xFFF0000 & to_send.RDLR) >> 16) * 0.05 -25) * 1.609;
    if (speed_kph < 0) {
      speed_kph = 0;
    }
    speed_kph = 20; //force it at 20 kph for debug
    speed_kph = (int)(speed_kph/0.04) & 0x1FFF;
    to_send.RDLR = (speed_kph | (speed_kph << 13) | (speed_kph << 26)) & 0xFFFFFFFF;
    to_send.RDHR = ((speed_kph  >> 6) | (speed_kph << 7) | (counter << 20)) & 0x00FFFFFF;
    int cksm = 0x76;
    cksm = (cksm + (to_send.RDLR & 0xFF) + ((to_send.RDLR >> 8) & 0xFF) + ((to_send.RDLR >> 16) & 0xFF) + ((to_send.RDLR >> 24) & 0xFF)) & 0xFF;
    cksm = (cksm + (to_send.RDHR & 0xFF) + ((to_send.RDHR >> 8) & 0xFF) + ((to_send.RDHR >> 16) & 0xFF) + ((to_send.RDHR >> 24) & 0xFF)) & 0xFF;
    to_send.RDHR = to_send.RDHR | (cksm << 24);
  }
  if (addr == 0x108 )
  {
    to_send.RIR = (0x109 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x308 )
  {
    to_send.RIR = (0x209 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x45 )
  {
    to_send.RIR = (0x219 << 21) + (addr_mask & (to_fwd->RIR | 1));
    to_send.RDLR = to_send.RDLR & 0xFFFF00FF;
    to_send.RDHR = to_send.RDHR & 0x00FFFFFF;
    int crc = add_tesla_crc(&to_send,7);
    to_send.RDHR = to_send.RDHR | (crc << 24);
  }
  if (addr == 0x148 )
  {
    to_send.RIR = (0x1A9 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x30A)
  {
    to_send.RIR = (0x2D9 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  can_send(&to_send, bus_num);
}

static int tesla_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd)
{

  int32_t addr = to_fwd->RIR >> 21;

  if (bus_num == 0)
  {

    //check all messages we need to also send to radar unmoddified
    /*if ((addr == 0x649 ) || (addr == 0x730 ) || (addr == 0x647 ) || (addr == 0x644 ) || (addr == 0x645 ) || (addr == 0x7DF ) || (addr == 0x64D ) || (addr == 0x64C ) || (addr == 0x643 ) || 
       (addr == 0x64E ) || (addr == 0x671 ) || (addr == 0x674 ) || (addr == 0x675 ) || (addr == 0x672 ) || (addr == 0x673 ) || (addr == 0x7F1 ) || (addr == 0x641 ) || (addr == 0x790 ) || (addr == 0x64B ) || 
       (addr == 0x64F ) || (addr == 0x71800 ) || (addr == 0x72B)) */
    /*if ((addr == 0x641) || (addr == 0x537))
    {
      //these messages are just forwarded with the same IDs
      tesla_fwd_to_radar_as_is(1, to_fwd);
    }*/

    //check all messages we need to also send to radar moddified
    //145 does not exist, we use 115 at the same frequency to trigger
    //175 does not exist, we use 118 at the same frequency to trigger and pass vehicle speed
    if ((addr == 0x405 ) || (addr == 0x398 ) || (addr == 0xE ) || (addr == 0x20A ) || (addr == 0x118 ) || (addr == 0x108 ) || (addr == 0x308 ) || 
    (addr == 0x115 ) || (addr == 0x45 ) || (addr == 0x148 ) || (addr == 0x30A)) 
    {
      tesla_fwd_to_radar_modded(1, to_fwd);
    }

    // change inhibit of GTW_epasControl
    if (addr == 0x101)
    {
      to_fwd->RDLR = to_fwd->RDLR | 0x4000; // 0x4000: WITH_ANGLE, 0xC000: WITH_BOTH (angle and torque)
      int checksum = (((to_fwd->RDLR & 0xFF00) >> 8) + (to_fwd->RDLR & 0xFF) + 2) & 0xFF;
      to_fwd->RDLR = to_fwd->RDLR & 0xFFFF;
      to_fwd->RDLR = to_fwd->RDLR + (checksum << 16);
      return 2;
    }

    // remove EPB_epasControl
    if (addr == 0x214)
    {
      return -1;
    }

    return 2; // Custom EPAS bus
  }

  if ((bus_num != 0) && (bus_num != 2)) {
    //everything but the radar data 0x300-0x3FF will be forwarded to can 0
    if ((addr > 0x300) && (addr <= 0x3FF)){ 
      return -1;
    }
    return 0;
  }

  if (bus_num == 2)
  {

    // remove GTW_epasControl in forwards
    if (addr == 0x101)
    {
      return -1;
    }

    // remove Pedal in forwards
    if ((addr == 0x520) || (addr == 0x521)) {
      return -1;
    }

    return 0; // Chassis CAN
  }
  return -1;
}

const safety_hooks tesla_hooks = {
    .init = tesla_init,
    .rx = tesla_rx_hook,
    .tx = tesla_tx_hook,
    .tx_lin = tesla_tx_lin_hook,
    .ignition = tesla_ign_hook,
    .fwd = tesla_fwd_hook,
};
