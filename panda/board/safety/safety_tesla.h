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
    {410., 92., 36.}};

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

/* <-- revB giraffe GPIO */
#include "../drivers/uja1023.h"

uint32_t tesla_ts_brakelight_on_last = 0;
const int32_t BRAKELIGHT_CLEAR_INTERVAL = 250000; //25ms; needs to be slower than the framerate difference between the DI_torque2 (~100Hz) and DI_state messages (~10hz).
const int32_t STW_MENU_BTN_HOLD_INTERVAL = 750000; //75ms, how long before we recognize the user is  holding this steering wheel button down

uint32_t stw_menu_btn_pressed_ts = 0;
int stw_menu_current_output_state = 0;
int stw_menu_btn_state_last = 0;
int stw_menu_output_flag = 0;
int high_beam_lever_state = 0;
/* revB giraffe GPIO --> */

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
  uint32_t ts = TIM2->CNT;

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
    /* <-- openpilot */
    // 6 bits starting at position 0
    int ap_lever_position = (to_push->RDLR & 0x3F);
    if (ap_lever_position == 2)
    { // pull forward
      // activate openpilot
      // TODO: uncomment the if to use double pull to activate
      //if (current_car_time <= time_at_last_stalk_pull + 1 && current_car_time != -1 && time_at_last_stalk_pull != -1) {
      controls_allowed = 1;
      //}
      time_at_last_stalk_pull = current_car_time;
    }
    else if (ap_lever_position == 1)
    { // push towards the back
      // deactivate openpilot
      controls_allowed = 0;
    }
    /* openpilot --> */
   
    /* <-- revB giraffe GPIO */
    int turn_signal_lever = (to_push->RDLR >> 16) & 0x3; //TurnIndLvr_Stat : 16|2@1+
    int stw_menu_button = (to_push->RDHR >> 5) & 0x1; //StW_Sw05_Psd : 37|1@1+
    high_beam_lever_state = (to_push->RDLR >> 18) & 0x3; //SG_ HiBmLvr_Stat : 18|2@1+
    
    //TurnIndLvr_Stat 3 "SNA" 2 "RIGHT" 1 "LEFT" 0 "IDLE" ;
    if (turn_signal_lever == 1)
    {
      //Left turn signal is on, turn on output pin 3
      set_uja1023_output_bits(1 << 3);
      //puts(" Left turn on!\n");
    }
    else
    {
      clear_uja1023_output_bits(1 << 3);
    }
    if (turn_signal_lever == 2)
    {
      //Right turn signal is on, turn on output pin 4
      set_uja1023_output_bits(1 << 4);
      //puts(" Right turn on!\n");
    }
    else
    {
      clear_uja1023_output_bits(1 << 4);
    }
    
    if (stw_menu_button == 1)
    {
      //menu button is pushed, if it wasn't last time, set the initial timestamp
      if (stw_menu_btn_state_last == 0)
      {
        stw_menu_btn_state_last = 1;
        stw_menu_btn_pressed_ts = ts;  
      }
      else
      {
        uint32_t stw_ts_elapsed = get_ts_elapsed(ts, stw_menu_btn_pressed_ts);
        if (stw_ts_elapsed > STW_MENU_BTN_HOLD_INTERVAL)
        {
          //user held the button, do stuff!
          if (stw_menu_current_output_state == 0 && stw_menu_output_flag == 0)
          {
            stw_menu_output_flag = 1;
            stw_menu_current_output_state = 1;
            //set_uja1023_output_bits(1 << 5);
            //puts("Menu Button held, setting output 5 HIGH\n");
          }
          else if (stw_menu_current_output_state == 1 && stw_menu_output_flag == 0)
          {
            stw_menu_output_flag = 1;
            stw_menu_current_output_state = 0;
            //clear_uja1023_output_bits(1 << 5);
            //puts("Menu Button held, setting output 5 LOW\n");
          }
        } //held
      }
    } //stw menu button pressed
    else if (stw_menu_button == 0)
    {
      stw_menu_output_flag = 0;
      stw_menu_btn_state_last = 0;
    }
    /* revB giraffe GPIO --> */
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
  /* revB giraffe GPIO --> */
  if (addr == 0x118)
  {
    int drive_state = (to_push->RDLR >> 12) & 0x7; //DI_gear : 12|3@1+
    int brake_pressed = (to_push->RDLR & 0x8000) >> 15;
    int tesla_speed_mph = ((((((to_push->RDLR >> 24) & 0x0F) << 8) + (( to_push->RDLR >> 16) & 0xFF)) * 0.05 -25));
    
    //if the car goes into reverse, set UJA1023 output pin 5 to high. If Drive, set pin 1 high
    //DI_gear 7 "DI_GEAR_SNA" 4 "DI_GEAR_D" 3 "DI_GEAR_N" 2 "DI_GEAR_R" 1 "DI_GEAR_P" 0 "DI_GEAR_INVALID" ;
    //UJA1023 pin0 is our output to the camera switcher

    if (drive_state == 2)
    {
      //reverse_state = 1;
      set_uja1023_output_bits(1 << 5);
      
      //if we're in reverse, we always want the rear camera up:
      set_uja1023_output_bits(1 << 0); //show rear camera
      //puts(" Got Reverse\n");
      
    }
    else
    {
      //reverse_state = 0;
      clear_uja1023_output_bits(1 << 5);
      
      //if we're in not in reverse and button state is 0, set output low (show front camera)
      if (stw_menu_current_output_state == 0)
      {
        clear_uja1023_output_bits(1 << 0); //show front camera
      }
      
      //if we're not in reverse and button state is 1, set the output high (show the rear camera)
      else
      {
        set_uja1023_output_bits(1 << 0); //show rear camera
      } 
    }
    
    if (drive_state == 4)
    {
      set_uja1023_output_bits(1 << 1);
      //puts(" Got Drive\n");
    }
    else
    {
      clear_uja1023_output_bits(1 << 1);
    }

    if (brake_pressed == 1)
    {
      //disable break cancel by commenting line below
      //controls_allowed = 0;
    
      set_uja1023_output_bits(1 << 2);
      //puts(" Brake on!\n");
      tesla_ts_brakelight_on_last = ts;
    }
    else
    {
      uint32_t ts_elapsed = get_ts_elapsed(ts, tesla_ts_brakelight_on_last);
      if (ts_elapsed > BRAKELIGHT_CLEAR_INTERVAL)
      {
        clear_uja1023_output_bits(1 << 2);
        //puts(" Brakelight off!\n");
      } 
    }
    /* revB giraffe GPIO --> */
    
    //get vehicle speed in m/2. Tesla gives MPH
    tesla_speed = (tesla_speed_mph*1.609/3.6);
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
  
  /* <-- revB giraffe GPIO */
  //BO_ 1001 DAS_bodyControls: 8 XXX
  if (addr == 0x3e9)
  {
    int high_beam_decision = (to_push->RDLR >> 10) & 0x3; //DAS_highLowBeamDecision : 10|2@1+
    // highLowBeamDecision:
    //0: Undecided (Car off)
    //1: Off
    //2: On
    //3: Auto High Beam is disabled
    //VAL_ 69 HiBmLvr_Stat 3 "SNA" 2 "HIBM_FLSH_ON_PSD" 1 "HIBM_ON_PSD" 0 "IDLE" ;

    //If the lever is in either high beam position and auto high beam is off or indicates highs should be on
    if ((high_beam_decision == 3 && (high_beam_lever_state == 2 || high_beam_lever_state == 1))
    || (high_beam_decision == 2 && (high_beam_lever_state == 2 || high_beam_lever_state == 1)))
    {
      //high beams are on. Set the output 6 high
      set_uja1023_output_bits(1 << 6);
      //puts("High Beam on!\n");
    } //high beams on!
    else
    {
      //high beams are off. Set the output 6 low
      clear_uja1023_output_bits(1 << 6);
      //puts("High Beam off!\n");
    } //high beams off
  } //DAS_bodyControls
  
  //BO_ 872 DI_state: 8 DI
  if (addr == 0x368)
  {
    int regen_brake_light = (to_push->RDLR >> 8) & 0x1; //DI_regenLight : 8|1@1+
    //if the car's brake lights are on, set pin 2 to high
    if (regen_brake_light == 1)
    {
      set_uja1023_output_bits(1 << 2);
      //puts(" Regen Brake Light on!\n");
      tesla_ts_brakelight_on_last = ts;
    }
    else
    {
      uint32_t ts_elapsed = get_ts_elapsed(ts, tesla_ts_brakelight_on_last);
      if (ts_elapsed > BRAKELIGHT_CLEAR_INTERVAL)
      {
        clear_uja1023_output_bits(1 << 2);
        //puts(" Brakelight off!\n");
      }
    }
  }
  /* revB giraffe GPIO --> */
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
  }
  if (addr == 0x398 )
  {
    to_send.RIR = (0x2A9 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x00E )
  {
    to_send.RIR = (0x199 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x175 )
  {
    to_send.RIR = (0x169 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x20A )
  {
    to_send.RIR = (0x159 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x145 )
  {
    to_send.RIR = (0x149 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x118 )
  {
    to_send.RIR = (0x119 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x108 )
  {
    to_send.RIR = (0x109 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x308 )
  {
    to_send.RIR = (0x209 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x115 )
  {
    to_send.RIR = (0x129 << 21) + (addr_mask & (to_fwd->RIR | 1));
  }
  if (addr == 0x045 )
  {
    to_send.RIR = (0x219 << 21) + (addr_mask & (to_fwd->RIR | 1));
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
    if ((addr == 0x649 ) || (addr == 0x730 ) || (addr == 0x647 ) || (addr == 0x644 ) || (addr == 0x645 ) || (addr == 0x7DF ) || (addr == 0x64D ) || (addr == 0x64C ) || (addr == 0x643 ) || 
       (addr == 0x64E ) || (addr == 0x671 ) || (addr == 0x674 ) || (addr == 0x675 ) || (addr == 0x672 ) || (addr == 0x673 ) || (addr == 0x7F1 ) || (addr == 0x641 ) || (addr == 0x790 ) || (addr == 0x64B ) || 
       (addr == 0x64F ) || (addr == 0x718 ) || (addr == 0x72B)) 
    {
      //these messages are just forwarded with the same IDs
      tesla_fwd_to_radar_as_is(1, to_fwd);
    }

    //check all messages we need to also send to radar moddified
    if ((addr == 0x405 ) || (addr == 0x398 ) || (addr == 0x00E ) || (addr == 0x175 ) || (addr == 0x20A ) || (addr == 0x145 ) || (addr == 0x118 ) || (addr == 0x108 ) || (addr == 0x308 ) || 
    (addr == 0x115 ) || (addr == 0x045 ) || (addr == 0x148 ) || (addr == 0x30A)) 
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
      return false;
    }

    return 2; // Custom EPAS bus
  }

  if (bus_num == 1) {
    //everything but the radar data 0x300-0x3FF will be forwarded to can 0
    if ((addr < 0x300) || (addr > 0x3FF)) {
      return 0;
    }
    return false;
  }

  if (bus_num == 2)
  {

    // remove GTW_epasControl in forwards
    if (addr == 0x101)
    {
      return false;
    }

    // remove Pedal in forwards
    if ((addr == 0x520) || (addr == 0x521)) {
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
