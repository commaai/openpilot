// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      regen paddle
//      accel rising edge
//      brake rising edge
//      brake > 0mph

// lateral limits
const int16_t MAX_ANGLE = 20; //Degrees

int tesla_brake_prev = 0;
int tesla_gas_prev = 0;
int tesla_speed = 0;
int current_car_time = -1;
int time_at_last_stalk_pull = -1;
int eac_status = 0;

// used when faking the epb epas enable signal
int epb_control_counter = 0;

int tesla_ignition_started = 0;

static void tesla_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
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
  
  // Detect gear in drive (start recording when in drive)
  //if (addr == 0x118 && bus_number == 0) {
  if (addr == 0x118) {
    // DI_torque2
    int current_gear = (to_push->RDLR & 0x7000) >> 12;
    tesla_ignition_started = current_gear > 1; //Park = 1. If out of park, we're "on."
  }
  
  // exit controls on brake press
  // DI_torque2::DI_brakePedal 0x118
  if (addr == 0x118) {
    // 1 bit at position 16
    if (((to_push->RDLR & 0x8000)) >> 15 == 1) {
      controls_allowed = 0;
    }
  }  
  
  // exit controls on EPAS error
  // EPAS_sysStatus::EPAS_eacStatus 0x370
  if (addr == 0x370) {
    // if EPAS_eacStatus is not 1 or 2, disable control
    eac_status = ((to_push->RDHR >> 21)) & 0x7;
    //if (eac_status != 1 && eac_status != 2) {
    //  controls_allowed = 0;
    //}
  }  
  
  
  // ACC steering wheel buttons
  //if (addr == 69) {
  //  int buttons = (to_push->RDHR >> 12) & 0x7;
  //  // res/set - enable, cancel button - disable
  //  if (buttons == 2 || buttons == 3) {
  //    controls_allowed = 1;
  //  } else if (buttons == 6) {
  //    controls_allowed = 0;
  //  }
  //}

  // exit controls on rising edge of brake press or on brake press when
  // speed > 0
  //if (addr == 241) {
  //  int brake = (to_push->RDLR & 0xFF00) >> 8;
  //  // Brake pedal's potentiometer returns near-zero reading
  //  // even when pedal is not pressed
  //  if (brake < 10) {
  //    brake = 0;
  //  }
  //  if (brake && (!tesla_brake_prev || tesla_speed)) {
  //     controls_allowed = 0;
  //  }
  //  tesla_brake_prev = brake;
  //}

}

// all commands: gas/regen, friction brake and steering
// if controls_allowed and no pedals pressed
//     allow all commands up to limit
// else
//     block all commands that produce actuation

static int tesla_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  uint32_t addr;
  int angle_raw;
  int angle_steer;

  // 1 allows the message through
  addr = to_send->RIR >> 21;
  
  // do not transmit CAN message if steering angle too high
  // DAS_steeringControl::DAS_steeringAngleRequest
  if (addr == 0x488) {
    angle_raw = to_send->RDLR & 0x7F;
    angle_steer = angle_raw / 10 - 1638.35;
    if ( (angle_steer > MAX_ANGLE) || (angle_steer < -MAX_ANGLE) ) {
      return 0;
    }
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
}

static int tesla_ign_hook() {
  return tesla_ignition_started;
}

static int tesla_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  
  if (bus_num == 0) {
    int32_t addr = to_fwd->RIR >> 21;
    
    // change inhibit of GTW_epasControl to WITH_BOTH
    if (addr == 0x101) {
      to_fwd->RDLR = to_fwd->RDLR | 0xC000;
      int checksum = (((to_fwd->RDLR & 0xFF00) >> 8) + (to_fwd->RDLR & 0xFF) + 2) & 0xFF;
      to_fwd->RDLR = to_fwd->RDLR & 0xFFFF;
      to_fwd->RDLR = to_fwd->RDLR + (checksum << 16);
    }     
    
    // now create a fake EPB_epasControl signal in order to enable control on the EPAS
    //CAN_FIFOMailBox_TypeDef tx_to_push_epb;
    //tx_to_push_epb.RDHR = to_fwd->RDHR;
    //tx_to_push_epb.RDLR = to_fwd->RDLR;
    //tx_to_push_epb.RDTR = to_fwd->RDTR;
    //tx_to_push_epb.RIR = to_fwd->RIR;
    //
    //tx_to_push_epb.RIR = tx_to_push_epb.RIR & ~(0x7FF<<21);
    //tx_to_push_epb.RIR = tx_to_push_epb.RIR | (0x214 << 21);
    //tx_to_push_epb.RDLR = 1 + (epb_control_counter << 8);
    //checksum = (((tx_to_push_epb.RDLR & 0xFF00) >> 8) + (tx_to_push_epb.RDLR & 0xFF) + 0x16) & 0xFF;
    //tx_to_push_epb.RDLR = tx_to_push_epb.RDLR & 0xFFFF;
    //tx_to_push_epb.RDLR = tx_to_push_epb.RDLR + (checksum << 16);          
    //
    //// send fake EPB_epasControl signal
    //can_send(tx_to_push_epb, 3);          
    //if (epb_control_counter >= 15) epb_control_counter = 0;
    //else epb_control_counter++;    

    return 3; // Custom EPAS bus
  }
  if (bus_num == 3) {
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

