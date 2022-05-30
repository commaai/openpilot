const int CHRYSLER_MAX_STEER = 261;
const int CHRYSLER_MAX_RT_DELTA = 112;        // max delta torque allowed for real time checks
const uint32_t CHRYSLER_RT_INTERVAL = 250000; // 250ms between real time checks
const int CHRYSLER_MAX_RATE_UP = 3;           //Must be double of limits set in op
const int CHRYSLER_MAX_RATE_DOWN = 3;         //Must be double of limits set in op
const int CHRYSLER_MAX_TORQUE_ERROR = 80;    // max torque cmd in excess of torque motor
const int CHRYSLER_GAS_THRSLD = 7.7;          // 7% more than 2m/s changed from wheel rpm to km/h 
const int CHRYSLER_STANDSTILL_THRSLD = 3.6;   // about 1m/s changed from wheel rpm to km/h 
const int RAM_MAX_STEER = 363; 
const int RAM_MAX_RT_DELTA = 182;             // since 2 x the rate up from chrsyler, 3x this also NEEDS CONFIRMED
const int RAM_MAX_RATE_UP = 14;               //Must be double of limits set in op
const int RAM_MAX_RATE_DOWN = 14;             //Must be double of limits set in op  
const int RAM_MAX_TORQUE_ERROR = 400;         // since 2 x the rate up from chrsyler, 3x this also NEEDS CONFIRMED


// Safety-relevant CAN messages for Chrysler/Jeep platforms
#define EPS_2                      544  // EPS driver input torque
#define ESP_1                      320  // Brake pedal and vehicle speed
#define ESP_8                      284  // Brake pedal and vehicle speed
#define ECM_5                      559  // Throttle position sensor
#define DAS_3                      500  // ACC engagement states from DASM
#define DAS_6                      678  // LKAS HUD and auto headlight control from DASM
#define LKAS_COMMAND               658  // LKAS controls from DASM 
#define Cruise_Control_Buttons     571  // Cruise control buttons

// Safety-relevant CAN messages for the 5th gen RAM (DT) platform
#define EPS_2_RAM                   49  // EPS driver input torque
#define ESP_1_RAM                  131  // Brake pedal and vehicle speed
#define ESP_8_RAM                  121  // Brake pedal and vehicle speed
#define ECM_5_RAM                  157  // Throttle position sensor
#define DAS_3_RAM                  153  // ACC engagement states from DASM
#define DAS_6_RAM                  250  // LKAS HUD and auto headlight control from DASM
#define LKAS_COMMAND_RAM           166  // LKAS controls from DASM 
#define Cruise_Control_Buttons_RAM 177  // Cruise control buttons
#define Center_Stack_2_RAM         650  // Center Stack buttons

// Safety-relevant CAN messages for the 5th gen RAM HD platform
#define DAS_6_HD                   629  // LKAS HUD and auto headlight control from DASM
#define LKAS_COMMAND_HD            630  // LKAS controls from DASM 
#define Cruise_Control_Buttons_HD  570  // Cruise control buttons
#define Center_Stack_2_HD          650  // Center Stack buttons

const CanMsg CHRYSLER_TX_MSGS[] = {{Cruise_Control_Buttons, 0, 3},{LKAS_COMMAND, 0, 6}, {DAS_6, 0, 8},
  {Cruise_Control_Buttons_RAM, 2, 3}, {LKAS_COMMAND_RAM, 0, 8}, {DAS_6_RAM, 0, 8},
  {Cruise_Control_Buttons_HD, 2, 3}, {LKAS_COMMAND_HD, 0, 8}, {DAS_6_HD, 0, 8}};

AddrCheckStruct chrysler_addr_checks[] = {
  {.msg = {{EPS_2, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 10000U}, {EPS_2_RAM, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 10000U}}},  // EPS module
  {.msg = {{ESP_1, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, {ESP_1_RAM, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}}},  // brake pressed
  {.msg = {{ESP_8, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, {ESP_8_RAM, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}}},  // vehicle Speed
  {.msg = {{ECM_5, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, {ECM_5_RAM, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}}}, // gas pedal
  {.msg = {{DAS_3, 2, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, {DAS_3_RAM, 2, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}}}, // forward cam ACC may need set to bus 0 to for jeep/pacifica
};
#define CHRYSLER_ADDR_CHECK_LEN (sizeof(chrysler_addr_checks) / sizeof(chrysler_addr_checks[0]))
addr_checks chrysler_rx_checks = {chrysler_addr_checks, CHRYSLER_ADDR_CHECK_LEN};

static uint32_t chrysler_get_checksum(CANPacket_t *to_push) {
  int checksum_byte = GET_LEN(to_push) - 1U;
  return (uint8_t)(GET_BYTE(to_push, checksum_byte));
}

static uint32_t chrysler_compute_checksum(CANPacket_t *to_push) {
  /* This function does not want the checksum byte in the input data.
  jeep chrysler canbus checksum from http://illmatics.com/Remote%20Car%20Hacking.pdf */
  uint8_t checksum = 0xFFU;
  int len = GET_LEN(to_push);
  for (int j = 0; j < (len - 1); j++) {
    uint8_t shift = 0x80U;
    uint8_t curr = (uint8_t)GET_BYTE(to_push, j);
    for (int i=0; i<8; i++) {
      uint8_t bit_sum = curr & shift;
      uint8_t temp_chk = checksum & 0x80U;
      if (bit_sum != 0U) {
        bit_sum = 0x1C;
        if (temp_chk != 0U) {
          bit_sum = 1;
        }
        checksum = checksum << 1;
        temp_chk = checksum | 1U;
        bit_sum ^= temp_chk;
      } else {
        if (temp_chk != 0U) {
          bit_sum = 0x1D;
        }
        checksum = checksum << 1;
        bit_sum ^= checksum;
      }
      checksum = bit_sum;
      shift = shift >> 1;
    }
  }
  return (uint8_t)(~checksum);
}

static uint8_t chrysler_get_counter(CANPacket_t *to_push) {
  // Well defined counter only for 8 bytes messages
  return (uint8_t)(GET_BYTE(to_push, 6) >> 4);
}

static int chrysler_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &chrysler_rx_checks,
                                 chrysler_get_checksum, chrysler_compute_checksum,
                                 chrysler_get_counter);
  
  if (valid) {
    int bus = GET_BUS(to_push);
    int addr = GET_ADDR(to_push);

    // Measured eps torque
    if ((bus == 0U) && ((addr == EPS_2) || (addr == EPS_2_RAM))) {
      int torque_meas_new = ((GET_BYTE(to_push, 4) & 0x7U) << 8) + GET_BYTE(to_push, 5) - 1024U;

      // update array of samples
      update_sample(&torque_meas, torque_meas_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if (((bus == 0U) && (addr == DAS_3)) || ((bus == 2) && ((addr == DAS_3) ||(addr == DAS_3_RAM)))) {
      int cruise_engaged = (GET_BYTE(to_push, 2) & 0x20U) == 0x20U;
      if (cruise_engaged && !cruise_engaged_prev) {
        controls_allowed = 1;
      }
      if (!cruise_engaged) {
        controls_allowed = 0;
      }
      cruise_engaged_prev = cruise_engaged;
    }

    // update speed
    if ((bus == 0U) && ((addr == ESP_8) || (addr == ESP_8_RAM))) {
      vehicle_speed = (((GET_BYTE(to_push, 4) & 0x3U) << 8) + GET_BYTE(to_push, 5))*0.0078125;
      vehicle_moving = (int)vehicle_speed > CHRYSLER_STANDSTILL_THRSLD;
    }

    // exit controls on rising edge of gas press
    if ((bus == 0U) && ((addr == ECM_5) || (addr == ECM_5_RAM))) {
      gas_pressed = (((GET_BYTE(to_push, 0) & 0x7FU)*0.4) >45) && ((int)vehicle_speed > CHRYSLER_GAS_THRSLD);
    }

    // exit controls on rising edge of brake press
    if ((bus == 0U) && ((addr == ESP_1) || (addr == ESP_1_RAM))) {
      brake_pressed = (GET_BYTE(to_push, 0) & 0xCU) == 0x4U;
      if (brake_pressed && (!brake_pressed_prev || vehicle_moving)) {
        controls_allowed = 0;
      }
      brake_pressed_prev = brake_pressed;
    }

    generic_rx_checks((bus == 0U) && ((addr == LKAS_COMMAND) || (addr == LKAS_COMMAND_RAM) || (addr == LKAS_COMMAND_HD)));
  }
  return valid;
}

static int chrysler_tx_hook(CANPacket_t *to_send, bool longitudinal_allowed) {
  UNUSED(longitudinal_allowed);

  int tx = 1;
  int addr = GET_ADDR(to_send);

  if (!msg_allowed(to_send, CHRYSLER_TX_MSGS, sizeof(CHRYSLER_TX_MSGS) / sizeof(CHRYSLER_TX_MSGS[0]))) {
    tx = 0;
  }

  // LKA STEER Chrysler/Jeep
  if ((addr == LKAS_COMMAND)) {
    int desired_torque = ((GET_BYTE(to_send, 0) & 0x7U) << 8) + GET_BYTE(to_send, 1) - 1024U;
    uint32_t ts = microsecond_timer_get();
    bool violation = 0;

    if (controls_allowed) {

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, CHRYSLER_MAX_STEER, -CHRYSLER_MAX_STEER);

      // *** torque rate limit check ***
      violation |= dist_to_meas_check(desired_torque, desired_torque_last,
        &torque_meas, CHRYSLER_MAX_RATE_UP, CHRYSLER_MAX_RATE_DOWN, CHRYSLER_MAX_TORQUE_ERROR);

      // used next time
      desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, rt_torque_last, CHRYSLER_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, ts_last);
      if (ts_elapsed > CHRYSLER_RT_INTERVAL) {
        rt_torque_last = desired_torque;
        ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!controls_allowed && (desired_torque != 0)) {
      violation = 1;
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (violation || !controls_allowed) {
      desired_torque_last = 0;
      rt_torque_last = 0;
      ts_last = ts;
    }

    if (violation) {
      tx = 0;
    }
  }

  // LKA STEER Ram
  if ((addr == LKAS_COMMAND_RAM) ||(addr == LKAS_COMMAND_HD)) {
    int desired_torque = ((GET_BYTE(to_send, 1) & 0x7U) << 8) + GET_BYTE(to_send, 2) - 1024U;
    uint32_t ts = TIM2->CNT;
    bool violation = 0;

    if (controls_allowed) {

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, RAM_MAX_STEER, -RAM_MAX_STEER);

      // *** torque rate limit check ***
      //violation |= dist_to_meas_check(desired_torque, desired_torque_last,
        //&torque_meas, RAM_MAX_RATE_UP, RAM_MAX_RATE_DOWN, RAM_MAX_TORQUE_ERROR);

      // used next time
      desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, rt_torque_last, RAM_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, ts_last);
      if (ts_elapsed > CHRYSLER_RT_INTERVAL) {
        rt_torque_last = desired_torque;
        ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!controls_allowed && (desired_torque != 0)) {
      violation = 1;
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (violation || !controls_allowed) {
      desired_torque_last = 0;
      rt_torque_last = 0;
      ts_last = ts;
    }

    if (violation) {
      tx = 0;
    }
  }

  

  // FORCE CANCEL: only the cancel button press is allowed
  if ((addr == Cruise_Control_Buttons) || (addr == Cruise_Control_Buttons_RAM) || (addr == Cruise_Control_Buttons_HD)) {
    if ((GET_BYTE(to_send, 0) != 1U) || ((GET_BYTE(to_send, 1) & 1U) == 1U)) {
      tx = 0;
    }
  }

  return tx;
}

static int chrysler_fwd_hook(int bus_num, CANPacket_t *to_fwd) {

  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);

  // forward CAN 0 -> 2 so stock LKAS camera sees messages
  if (bus_num == 0U && (addr != Center_Stack_2_RAM)) {//Ram and HD share the same
    bus_fwd = 2;
  }

  // forward all messages from camera except LKAS_COMMAND and LKAS_HUD
  if ((bus_num == 2U) && (addr != LKAS_COMMAND) && (addr != DAS_6) 
    && (addr != LKAS_COMMAND_RAM) && (addr != DAS_6_RAM)
    && (addr != LKAS_COMMAND_HD) && (addr != DAS_6_HD)){
    bus_fwd = 0;
  }

  return bus_fwd;
}

static const addr_checks* chrysler_init(uint16_t param) {
  UNUSED(param);
  controls_allowed = false;
  relay_malfunction_reset();
  return &chrysler_rx_checks;
}

const safety_hooks chrysler_hooks = {
  .init = chrysler_init,
  .rx = chrysler_rx_hook,
  .tx = chrysler_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = chrysler_fwd_hook,
};
