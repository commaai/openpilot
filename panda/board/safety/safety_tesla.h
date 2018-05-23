// board enforces
//   in-state
//      accel set/resume
//   out-state
//      cancel button
//      regen paddle
//      accel rising edge
//      brake rising edge
//      brake > 0mph

int tesla_brake_prev = 0;
int tesla_gas_prev = 0;
int tesla_speed = 0;

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
  
  // Detect gear in drive (start recording when in drive)
  //if (addr == 0x118 && bus_number == 0) {
  if (addr == 0x118) {
    // DI_torque2
    int current_gear = (to_push->RDLR & 0x7000) >> 12;
    tesla_ignition_started = current_gear > 1; //Park = 1. If out of park, we're "on."
    //tesla_ignition_started = 1; //TEMPORARY TEST
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

  //uint32_t addr;
  //if (to_send->RIR & 4) {
  //  // Extended
  //  addr = to_send->RIR >> 3;
  //} else {
  //  // Normal
  //  addr = to_send->RIR >> 21;
  //}

  // 1 allows the message through
  return false;
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

