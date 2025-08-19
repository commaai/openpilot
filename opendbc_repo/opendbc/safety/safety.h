#pragma once

#include "opendbc/safety/helpers.h"
#include "opendbc/safety/lateral.h"
#include "opendbc/safety/longitudinal.h"
#include "opendbc/safety/safety_declarations.h"
#include "opendbc/safety/board/can.h"

// all the safety modes
#include "opendbc/safety/modes/defaults.h"
#include "opendbc/safety/modes/honda.h"
#include "opendbc/safety/modes/toyota.h"
#include "opendbc/safety/modes/tesla.h"
#include "opendbc/safety/modes/gm.h"
#include "opendbc/safety/modes/ford.h"
#include "opendbc/safety/modes/hyundai.h"
#include "opendbc/safety/modes/chrysler.h"
#include "opendbc/safety/modes/rivian.h"
#include "opendbc/safety/modes/subaru.h"
#include "opendbc/safety/modes/subaru_preglobal.h"
#include "opendbc/safety/modes/mazda.h"
#include "opendbc/safety/modes/nissan.h"
#include "opendbc/safety/modes/volkswagen_mqb.h"
#include "opendbc/safety/modes/volkswagen_pq.h"
#include "opendbc/safety/modes/elm327.h"
#include "opendbc/safety/modes/body.h"
#include "opendbc/safety/modes/psa.h"

// CAN-FD only safety modes
#ifdef CANFD
#include "opendbc/safety/modes/hyundai_canfd.h"
#endif

uint32_t GET_BYTES(const CANPacket_t *msg, int start, int len) {
  uint32_t ret = 0U;
  for (int i = 0; i < len; i++) {
    const uint32_t shift = i * 8;
    ret |= (((uint32_t)msg->data[start + i]) << shift);
  }
  return ret;
}

const int MAX_WRONG_COUNTERS = 5;

// This can be set by the safety hooks
bool controls_allowed = false;
bool relay_malfunction = false;
bool gas_pressed = false;
bool gas_pressed_prev = false;
bool brake_pressed = false;
bool brake_pressed_prev = false;
bool regen_braking = false;
bool regen_braking_prev = false;
bool steering_disengage;
bool steering_disengage_prev;
bool cruise_engaged_prev = false;
struct sample_t vehicle_speed;
bool vehicle_moving = false;
bool acc_main_on = false;  // referred to as "ACC off" in ISO 15622:2018
int cruise_button_prev = 0;
bool safety_rx_checks_invalid = false;

// for safety modes with torque steering control
int desired_torque_last = 0;       // last desired steer torque
int rt_torque_last = 0;            // last desired torque for real time check
int valid_steer_req_count = 0;     // counter for steer request bit matching non-zero torque
int invalid_steer_req_count = 0;   // counter to allow multiple frames of mismatching torque request bit
struct sample_t torque_meas;       // last 6 motor torques produced by the eps
struct sample_t torque_driver;     // last 6 driver torques measured
uint32_t ts_torque_check_last = 0;
uint32_t ts_steer_req_mismatch_last = 0;  // last timestamp steer req was mismatched with torque

// state for controls_allowed timeout logic
bool heartbeat_engaged = false;             // openpilot enabled, passed in heartbeat USB command
uint32_t heartbeat_engaged_mismatches = 0;  // count of mismatches between heartbeat_engaged and controls_allowed

// for safety modes with angle steering control
uint32_t rt_angle_msgs = 0;
uint32_t ts_angle_check_last = 0;
int desired_angle_last = 0;
struct sample_t angle_meas;         // last 6 steer angles/curvatures


int alternative_experience = 0;

// time since safety mode has been changed
uint32_t safety_mode_cnt = 0U;

uint16_t current_safety_mode = SAFETY_SILENT;
uint16_t current_safety_param = 0;
static const safety_hooks *current_hooks = &nooutput_hooks;
safety_config current_safety_config;

static void generic_rx_checks(void);
static void stock_ecu_check(bool stock_ecu_detected);

static bool is_msg_valid(RxCheck addr_list[], int index) {
  bool valid = true;
  if (index != -1) {
    if (!addr_list[index].status.valid_checksum || !addr_list[index].status.valid_quality_flag || (addr_list[index].status.wrong_counters >= MAX_WRONG_COUNTERS)) {
      valid = false;
      controls_allowed = false;
    }
  }
  return valid;
}

static int get_addr_check_index(const CANPacket_t *msg, RxCheck addr_list[], const int len) {
  int addr = msg->addr;
  int length = GET_LEN(msg);

  int index = -1;
  for (int i = 0; i < len; i++) {
    // if multiple msgs are allowed, determine which one is present on the bus
    if (!addr_list[i].status.msg_seen) {
      for (uint8_t j = 0U; (j < MAX_ADDR_CHECK_MSGS) && (addr_list[i].msg[j].addr != 0); j++) {
        if ((addr == addr_list[i].msg[j].addr) && (msg->bus == addr_list[i].msg[j].bus) &&
              (length == addr_list[i].msg[j].len)) {
          addr_list[i].status.index = j;
          addr_list[i].status.msg_seen = true;
          break;
        }
      }
    }

    if (addr_list[i].status.msg_seen) {
      int idx = addr_list[i].status.index;
      if ((addr == addr_list[i].msg[idx].addr) && (msg->bus == addr_list[i].msg[idx].bus) &&
          (length == addr_list[i].msg[idx].len)) {
        index = i;
        break;
      }
    }
  }
  return index;
}

static void update_addr_timestamp(RxCheck addr_list[], int index) {
  if (index != -1) {
    uint32_t ts = microsecond_timer_get();
    addr_list[index].status.last_timestamp = ts;
  }
}

static void update_counter(RxCheck addr_list[], int index, uint8_t counter) {
  if (index != -1) {
    uint8_t expected_counter = (addr_list[index].status.last_counter + 1U) % (addr_list[index].msg[addr_list[index].status.index].max_counter + 1U);
    addr_list[index].status.wrong_counters += (expected_counter == counter) ? -1 : 1;
    addr_list[index].status.wrong_counters = CLAMP(addr_list[index].status.wrong_counters, 0, MAX_WRONG_COUNTERS);
    addr_list[index].status.last_counter = counter;
  }
}

static bool rx_msg_safety_check(const CANPacket_t *msg,
                                const safety_config *cfg,
                                const safety_hooks *safety_hooks) {

  int index = get_addr_check_index(msg, cfg->rx_checks, cfg->rx_checks_len);
  update_addr_timestamp(cfg->rx_checks, index);

  if (index != -1) {
    // checksum check
    if ((safety_hooks->get_checksum != NULL) && (safety_hooks->compute_checksum != NULL) && !cfg->rx_checks[index].msg[cfg->rx_checks[index].status.index].ignore_checksum) {
      uint32_t checksum = safety_hooks->get_checksum(msg);
      uint32_t checksum_comp = safety_hooks->compute_checksum(msg);
      cfg->rx_checks[index].status.valid_checksum = checksum_comp == checksum;
    } else {
      cfg->rx_checks[index].status.valid_checksum = cfg->rx_checks[index].msg[cfg->rx_checks[index].status.index].ignore_checksum;
    }

    // counter check
    if ((safety_hooks->get_counter != NULL) && (cfg->rx_checks[index].msg[cfg->rx_checks[index].status.index].max_counter > 0U)) {
      uint8_t counter = safety_hooks->get_counter(msg);
      update_counter(cfg->rx_checks, index, counter);
    } else {
      cfg->rx_checks[index].status.wrong_counters = cfg->rx_checks[index].msg[cfg->rx_checks[index].status.index].ignore_counter ? 0 : MAX_WRONG_COUNTERS;
    }

    // quality flag check
    if ((safety_hooks->get_quality_flag_valid != NULL) && !cfg->rx_checks[index].msg[cfg->rx_checks[index].status.index].ignore_quality_flag) {
      cfg->rx_checks[index].status.valid_quality_flag = safety_hooks->get_quality_flag_valid(msg);
    } else {
      cfg->rx_checks[index].status.valid_quality_flag = cfg->rx_checks[index].msg[cfg->rx_checks[index].status.index].ignore_quality_flag;
    }
  }
  return is_msg_valid(cfg->rx_checks, index);
}

bool safety_rx_hook(const CANPacket_t *msg) {
  bool controls_allowed_prev = controls_allowed;

  bool valid = rx_msg_safety_check(msg, &current_safety_config, current_hooks);
  bool whitelisted = get_addr_check_index(msg, current_safety_config.rx_checks, current_safety_config.rx_checks_len) != -1;
  if (valid && whitelisted) {
    current_hooks->rx(msg);
  }

  // Handles gas, brake, and regen paddle
  generic_rx_checks();

  // the relay malfunction hook runs on all incoming rx messages.
  // check all applicable tx msgs for liveness on sending bus.
  // used to detect a relay malfunction or control messages from disabled ECUs like the radar
  const int addr = msg->addr;
  for (int i = 0; i < current_safety_config.tx_msgs_len; i++) {
    const CanMsg *m = &current_safety_config.tx_msgs[i];
    if (m->check_relay) {
      stock_ecu_check((m->addr == addr) && (m->bus == msg->bus));
    }
  }

  // reset mismatches on rising edge of controls_allowed to avoid rare race condition
  if (controls_allowed && !controls_allowed_prev) {
    heartbeat_engaged_mismatches = 0;
  }

  return valid;
}

static bool tx_msg_safety_check(const CANPacket_t *msg, const CanMsg msg_list[], int len) {
  int addr = msg->addr;
  int length = GET_LEN(msg);

  bool whitelisted = false;
  for (int i = 0; i < len; i++) {
    if ((addr == msg_list[i].addr) && (msg->bus == msg_list[i].bus) && (length == msg_list[i].len)) {
      whitelisted = true;
      break;
    }
  }
  return whitelisted;
}

bool safety_tx_hook(CANPacket_t *msg) {
  bool whitelisted = tx_msg_safety_check(msg, current_safety_config.tx_msgs, current_safety_config.tx_msgs_len);
  if ((current_safety_mode == SAFETY_ALLOUTPUT) || (current_safety_mode == SAFETY_ELM327)) {
    whitelisted = true;
  }

  bool safety_allowed = false;
  if (whitelisted) {
    safety_allowed = current_hooks->tx(msg);
  }

  return !relay_malfunction && whitelisted && safety_allowed;
}

static int get_fwd_bus(int bus_num) {
  int destination_bus;
  if (bus_num == 0) {
    destination_bus = 2;
  } else if (bus_num == 2) {
    destination_bus = 0;
  } else {
    destination_bus = -1;
  }
  return destination_bus;
}

int safety_fwd_hook(int bus_num, int addr) {
  bool blocked = relay_malfunction || current_safety_config.disable_forwarding;

  // Block messages that are being checked for relay malfunctions. Safety modes can opt out of this
  // in the case of selective AEB forwarding
  const int destination_bus = get_fwd_bus(bus_num);
  if (!blocked) {
    for (int i = 0; i < current_safety_config.tx_msgs_len; i++) {
      const CanMsg *m = &current_safety_config.tx_msgs[i];
      if (m->check_relay && !m->disable_static_blocking && (m->addr == addr) && (m->bus == (unsigned int)destination_bus)) {
        blocked = true;
        break;
      }
    }
  }

  if (!blocked && (current_hooks->fwd != NULL)) {
    blocked = current_hooks->fwd(bus_num, addr);
  }

  return blocked ? -1 : destination_bus;
}

// Given a CRC-8 poly, generate a static lookup table to use with a fast CRC-8
// algorithm. Called at init time for safety modes using CRC-8.
void gen_crc_lookup_table_8(uint8_t poly, uint8_t crc_lut[]) {
  for (uint16_t i = 0U; i <= 0xFFU; i++) {
    uint8_t crc = (uint8_t)i;
    for (int j = 0; j < 8; j++) {
      if ((crc & 0x80U) != 0U) {
        crc = (uint8_t)((crc << 1) ^ poly);
      } else {
        crc <<= 1;
      }
    }
    crc_lut[i] = crc;
  }
}

#ifdef CANFD
void gen_crc_lookup_table_16(uint16_t poly, uint16_t crc_lut[]) {
  for (uint16_t i = 0; i < 256U; i++) {
    uint16_t crc = i << 8U;
    for (uint16_t j = 0; j < 8U; j++) {
      if ((crc & 0x8000U) != 0U) {
        crc = (uint16_t)((crc << 1) ^ poly);
      } else {
        crc <<= 1;
      }
    }
    crc_lut[i] = crc;
  }
}
#endif

// 1Hz safety function called by main. Now just a check for lagging safety messages
void safety_tick(const safety_config *cfg) {
  const uint8_t MAX_MISSED_MSGS = 10U;
  bool rx_checks_invalid = false;
  uint32_t ts = microsecond_timer_get();
  if (cfg != NULL) {
    for (int i=0; i < cfg->rx_checks_len; i++) {
      uint32_t elapsed_time = get_ts_elapsed(ts, cfg->rx_checks[i].status.last_timestamp);
      // lag threshold is max of: 1s and MAX_MISSED_MSGS * expected timestep.
      // Quite conservative to not risk false triggers.
      // 2s of lag is worse case, since the function is called at 1Hz
      uint32_t timestep = 1e6 / cfg->rx_checks[i].msg[cfg->rx_checks[i].status.index].frequency;
      bool lagging = elapsed_time > MAX(timestep * MAX_MISSED_MSGS, 1e6);
      cfg->rx_checks[i].status.lagging = lagging;
      if (lagging) {
        controls_allowed = false;
      }

      if (lagging || !is_msg_valid(cfg->rx_checks, i)) {
        rx_checks_invalid = true;
      }
    }
  }

  safety_rx_checks_invalid = rx_checks_invalid;
}

static void relay_malfunction_set(void) {
  relay_malfunction = true;
  fault_occurred(FAULT_RELAY_MALFUNCTION);
}

static void generic_rx_checks(void) {
  gas_pressed_prev = gas_pressed;

  // exit controls on rising edge of brake press
  if (brake_pressed && (!brake_pressed_prev || vehicle_moving)) {
    controls_allowed = false;
  }
  brake_pressed_prev = brake_pressed;

  // exit controls on rising edge of regen paddle
  if (regen_braking && (!regen_braking_prev || vehicle_moving)) {
    controls_allowed = false;
  }
  regen_braking_prev = regen_braking;

  // exit controls on rising edge of steering override/disengage
  if (steering_disengage && !steering_disengage_prev) {
    controls_allowed = false;
  }
  steering_disengage_prev = steering_disengage;
}

static void stock_ecu_check(bool stock_ecu_detected) {
  // allow 1s of transition timeout after relay changes state before assessing malfunctioning
  const uint32_t RELAY_TRNS_TIMEOUT = 1U;

  // check if stock ECU is on bus broken by car harness
  if ((safety_mode_cnt > RELAY_TRNS_TIMEOUT) && stock_ecu_detected) {
    relay_malfunction_set();
  }
}

static void relay_malfunction_reset(void) {
  relay_malfunction = false;
  fault_recovered(FAULT_RELAY_MALFUNCTION);
}

// resets values and min/max for sample_t struct
static void reset_sample(struct sample_t *sample) {
  for (int i = 0; i < MAX_SAMPLE_VALS; i++) {
    sample->values[i] = 0;
  }
  update_sample(sample, 0);
}

int set_safety_hooks(uint16_t mode, uint16_t param) {
  const safety_hook_config safety_hook_registry[] = {
    {SAFETY_SILENT, &nooutput_hooks},
    {SAFETY_HONDA_NIDEC, &honda_nidec_hooks},
    {SAFETY_TOYOTA, &toyota_hooks},
    {SAFETY_ELM327, &elm327_hooks},
    {SAFETY_GM, &gm_hooks},
    {SAFETY_HONDA_BOSCH, &honda_bosch_hooks},
    {SAFETY_HYUNDAI, &hyundai_hooks},
    {SAFETY_CHRYSLER, &chrysler_hooks},
    {SAFETY_SUBARU, &subaru_hooks},
    {SAFETY_VOLKSWAGEN_MQB, &volkswagen_mqb_hooks},
    {SAFETY_NISSAN, &nissan_hooks},
    {SAFETY_NOOUTPUT, &nooutput_hooks},
    {SAFETY_HYUNDAI_LEGACY, &hyundai_legacy_hooks},
    {SAFETY_MAZDA, &mazda_hooks},
    {SAFETY_BODY, &body_hooks},
    {SAFETY_FORD, &ford_hooks},
    {SAFETY_RIVIAN, &rivian_hooks},
    {SAFETY_TESLA, &tesla_hooks},
#ifdef CANFD
    {SAFETY_HYUNDAI_CANFD, &hyundai_canfd_hooks},
#endif
#ifdef ALLOW_DEBUG
    {SAFETY_PSA, &psa_hooks},
    {SAFETY_SUBARU_PREGLOBAL, &subaru_preglobal_hooks},
    {SAFETY_VOLKSWAGEN_PQ, &volkswagen_pq_hooks},
    {SAFETY_ALLOUTPUT, &alloutput_hooks},
#endif
  };

  // reset state set by safety mode
  safety_mode_cnt = 0U;
  relay_malfunction = false;
  gas_pressed = false;
  gas_pressed_prev = false;
  brake_pressed = false;
  brake_pressed_prev = false;
  regen_braking = false;
  regen_braking_prev = false;
  steering_disengage = false;
  steering_disengage_prev = false;
  cruise_engaged_prev = false;
  vehicle_moving = false;
  acc_main_on = false;
  cruise_button_prev = 0;
  desired_torque_last = 0;
  rt_torque_last = 0;
  rt_angle_msgs = 0;
  ts_angle_check_last = 0;
  desired_angle_last = 0;
  ts_torque_check_last = 0;
  ts_steer_req_mismatch_last = 0;
  valid_steer_req_count = 0;
  invalid_steer_req_count = 0;

  // reset samples
  reset_sample(&vehicle_speed);
  reset_sample(&torque_meas);
  reset_sample(&torque_driver);
  reset_sample(&angle_meas);

  controls_allowed = false;
  relay_malfunction_reset();
  safety_rx_checks_invalid = false;

  current_safety_config.rx_checks = NULL;
  current_safety_config.rx_checks_len = 0;
  current_safety_config.tx_msgs = NULL;
  current_safety_config.tx_msgs_len = 0;
  current_safety_config.disable_forwarding = false;

  int set_status = -1;  // not set
  int hook_config_count = sizeof(safety_hook_registry) / sizeof(safety_hook_config);
  for (int i = 0; i < hook_config_count; i++) {
    if (safety_hook_registry[i].id == mode) {
      current_hooks = safety_hook_registry[i].hooks;
      current_safety_mode = mode;
      current_safety_param = param;
      set_status = 0;  // set
    }
  }
  if ((set_status == 0) && (current_hooks->init != NULL)) {
    safety_config cfg = current_hooks->init(param);
    current_safety_config.rx_checks = cfg.rx_checks;
    current_safety_config.rx_checks_len = cfg.rx_checks_len;
    current_safety_config.tx_msgs = cfg.tx_msgs;
    current_safety_config.tx_msgs_len = cfg.tx_msgs_len;
    current_safety_config.disable_forwarding = cfg.disable_forwarding;
    // reset all dynamic fields in addr struct
    for (int j = 0; j < current_safety_config.rx_checks_len; j++) {
      current_safety_config.rx_checks[j].status = (RxStatus){0};
    }
  }
  return set_status;
}

// convert a trimmed integer to signed 32 bit int
int to_signed(int d, int bits) {
  int d_signed = d;
  int max_value = (1 << MAX((bits - 1), 0));
  if (d >= max_value) {
    d_signed = d - (1 << MAX(bits, 0));
  }
  return d_signed;
}

// given a new sample, update the sample_t struct
void update_sample(struct sample_t *sample, int sample_new) {
  for (int i = MAX_SAMPLE_VALS - 1; i > 0; i--) {
    sample->values[i] = sample->values[i-1];
  }
  sample->values[0] = sample_new;

  // get the minimum and maximum measured samples
  sample->min = sample->values[0];
  sample->max = sample->values[0];
  for (int i = 1; i < MAX_SAMPLE_VALS; i++) {
    if (sample->values[i] < sample->min) {
      sample->min = sample->values[i];
    }
    if (sample->values[i] > sample->max) {
      sample->max = sample->values[i];
    }
  }
}

int ROUND(float val) {
  return val + ((val > 0.0) ? 0.5 : -0.5);
}

void pcm_cruise_check(bool cruise_engaged) {
  // Enter controls on rising edge of stock ACC, exit controls if stock ACC disengages
  if (!cruise_engaged) {
    controls_allowed = false;
  }
  if (cruise_engaged && !cruise_engaged_prev) {
    controls_allowed = true;
  }
  cruise_engaged_prev = cruise_engaged;
}

void speed_mismatch_check(const float speed_2) {
  // Disable controls if speeds from two sources are too far apart.
  // For safety modes that use speed to adjust torque or angle limits
  const float MAX_SPEED_DELTA = 2.0;  // m/s
  bool is_invalid_speed = ABS(speed_2 - ((float)vehicle_speed.values[0] / VEHICLE_SPEED_FACTOR)) > MAX_SPEED_DELTA;
  if (is_invalid_speed) {
    controls_allowed = false;
  }
}
