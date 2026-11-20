#pragma once

#include "opendbc/safety/helpers.h"
#include "opendbc/safety/lateral.h"
#include "opendbc/safety/longitudinal.h"
#include "opendbc/safety/declarations.h"
#include "opendbc/safety/can.h"

// all the safety modes
#include "opendbc/safety/modes/defaults.h"
#include "opendbc/safety/modes/honda.h"
#include "opendbc/safety/modes/toyota.h"
#include "opendbc/safety/modes/tesla.h"
#include "opendbc/safety/modes/gm.h"
#include "opendbc/safety/modes/ford.h"
#include "opendbc/safety/modes/hyundai.h"
#include "opendbc/safety/modes/chrysler.h"
#include "opendbc/safety/modes/chrysler_cusw.h"
#include "opendbc/safety/modes/rivian.h"
#include "opendbc/safety/modes/mg.h"
#include "opendbc/safety/modes/subaru.h"
#include "opendbc/safety/modes/subaru_preglobal.h"
#include "opendbc/safety/modes/mazda.h"
#include "opendbc/safety/modes/nissan.h"
#include "opendbc/safety/modes/volkswagen_mlb.h"
#include "opendbc/safety/modes/volkswagen_mqb.h"
#include "opendbc/safety/modes/volkswagen_meb.h"
#include "opendbc/safety/modes/volkswagen_pq.h"
#include "opendbc/safety/modes/elm327.h"
#include "opendbc/safety/modes/body.h"
#include "opendbc/safety/modes/psa.h"
#include "opendbc/safety/modes/byd.h"
#include "opendbc/safety/modes/hyundai_canfd.h"

uint32_t GET_BYTES_LE(const CANPacket_t *msg, int start, int len) {
  uint32_t ret = 0U;
  for (int i = 0; i < len; i++) {
    const uint32_t shift = i * 8;
    ret |= (((uint32_t)msg->data[start + i]) << shift);
  }
  return ret;
}

// Read up to eight bytes in little-endian order, without alignment assumptions.
uint64_t GET_BYTES_64_LE(const CANPacket_t *msg, int start, int len) {
  uint64_t ret = 0U;
  for (int i = 0; i < len; i++) {
    const uint32_t shift = i * 8;
    ret |= ((uint64_t)msg->data[start + i]) << shift;
  }
  return ret;
}

const int MAX_WRONG_COUNTERS = 5;

// UM2331 CPU_SM_2 and RAM_SM_3: rx, tx, fwd and tick run on two copies of the state and config and compare the decisions, the tick also compares the copies
SafetyCopy safety_a = {.setup = {.current_hooks = &nooutput_hooks}};
SafetyCopy safety_b = {.setup = {.current_hooks = &nooutput_hooks}};
SafetyCopy *current_copy = &safety_a;
uint32_t safety_ts;
bool safety_redundancy_fault = false;

static void generic_rx_checks(void);
static void stock_ecu_check(bool stock_ecu_detected);

static bool safety_states_equal(void) {
  const uint8_t *a = (const uint8_t *)&safety_a;
  const uint8_t *b = (const uint8_t *)&safety_b;
  bool equal = true;
  for (uint32_t i = 0U; i < sizeof(SafetyCopy); i++) {
    if (a[i] != b[i]) {
      equal = false;
    }
  }
  return equal;
}

// TODO: block on a mismatch
static bool safety_redundancy_check(bool same) {
  if (!same) {
    safety_b = safety_a;
    safety_redundancy_fault = true;
  }
  return same;
}

static bool is_msg_valid(int index) {
  bool valid = true;
  if (index != -1) {
    const RxStatus *status = &safety.rx_status[index];
    if (!status->valid_checksum || !status->valid_quality_flag || (status->wrong_counters >= MAX_WRONG_COUNTERS)) {
      valid = false;
      safety.controls_allowed = false;
    }
  }
  return valid;
}

static int get_addr_check_index(const CANPacket_t *msg, const RxCheck addr_list[], const int len) {
  int index = -1;
  for (int i = 0; i < len; i++) {
    RxStatus *status = &safety.rx_status[i];
    // if multiple msgs are allowed, determine which one is present on the bus
    if (!status->msg_seen) {
      for (uint8_t j = 0U; (j < MAX_ADDR_CHECK_MSGS) && (addr_list[i].msg[j].addr != 0U); j++) {
        if (msg_matches(msg, addr_list[i].msg[j].addr, addr_list[i].msg[j].bus, addr_list[i].msg[j].len)) {
          status->index = j;
          status->msg_seen = true;
          break;
        }
      }
    }

    if (status->msg_seen) {
      const CanMsgCheck *m = &addr_list[i].msg[status->index];
      if (msg_matches(msg, m->addr, m->bus, m->len)) {
        index = i;
        break;
      }
    }
  }
  return index;
}

static void update_counter(RxStatus *status, const CanMsgCheck *m, uint8_t counter) {
  uint8_t expected_counter = (status->last_counter + 1U) % (m->max_counter + 1U);
  status->wrong_counters += (expected_counter == counter) ? -1 : 1;
  status->wrong_counters = SAFETY_CLAMP(status->wrong_counters, 0, MAX_WRONG_COUNTERS);
  status->last_counter = counter;
}

static bool rx_msg_safety_check(const CANPacket_t *msg,
                                const safety_config *cfg,
                                const safety_hooks *safety_hooks) {

  int index = get_addr_check_index(msg, cfg->rx_checks, cfg->rx_checks_len);

  if (index != -1) {
    RxStatus *status = &safety.rx_status[index];
    const CanMsgCheck *m = &cfg->rx_checks[index].msg[status->index];
    status->last_timestamp = safety_ts;

    // checksum check
    if ((safety_hooks->get_checksum != NULL) && (safety_hooks->compute_checksum != NULL) && !m->ignore_checksum) {
      uint32_t checksum = safety_hooks->get_checksum(msg);
      uint32_t checksum_comp = safety_hooks->compute_checksum(msg);
      status->valid_checksum = checksum_comp == checksum;
    } else {
      status->valid_checksum = m->ignore_checksum;
    }

    // counter check
    if ((safety_hooks->get_counter != NULL) && (m->max_counter > 0U)) {
      uint8_t counter = safety_hooks->get_counter(msg);
      update_counter(status, m, counter);
    } else {
      status->wrong_counters = m->ignore_counter ? 0 : MAX_WRONG_COUNTERS;
    }

    // quality flag check
    if ((safety_hooks->get_quality_flag_valid != NULL) && !m->ignore_quality_flag) {
      status->valid_quality_flag = safety_hooks->get_quality_flag_valid(msg);
    } else {
      status->valid_quality_flag = m->ignore_quality_flag;
    }
  }
  return is_msg_valid(index);
}

static bool rx_hook_copy(const CANPacket_t *msg) {
  bool controls_allowed_prev = safety.controls_allowed;

  bool valid = rx_msg_safety_check(msg, &safety_setup.current_safety_config, safety_setup.current_hooks);
  bool whitelisted = get_addr_check_index(msg, safety_setup.current_safety_config.rx_checks, safety_setup.current_safety_config.rx_checks_len) != -1;
  if (valid && whitelisted) {
    safety_setup.current_hooks->rx(msg);
  }

  // Handles gas, brake, and regen paddle
  generic_rx_checks();

  // the relay malfunction hook runs on all incoming rx messages.
  // check all applicable tx msgs for liveness on sending bus.
  // used to detect a relay malfunction or control messages from disabled ECUs like the radar
  const int addr = msg->addr;
  for (int i = 0; i < safety_setup.current_safety_config.tx_msgs_len; i++) {
    const CanMsg *m = &safety_setup.current_safety_config.tx_msgs[i];
    if (m->check_relay) {
      stock_ecu_check((m->addr == addr) && (m->bus == msg->bus));
    }
  }

  // reset mismatches on rising edge of controls_allowed to avoid rare race condition
  if (safety.controls_allowed && !controls_allowed_prev) {
    safety.heartbeat_engaged_mismatches = 0;
  }

  return valid;
}

static bool tx_msg_safety_check(const CANPacket_t *msg, const CanMsg msg_list[], int len) {
  bool whitelisted = false;
  for (int i = 0; i < len; i++) {
    if (msg_matches(msg, msg_list[i].addr, msg_list[i].bus, msg_list[i].len)) {
      whitelisted = true;
      break;
    }
  }
  return whitelisted;
}

static bool tx_hook_copy(CANPacket_t *msg) {
  bool whitelisted = tx_msg_safety_check(msg, safety_setup.current_safety_config.tx_msgs, safety_setup.current_safety_config.tx_msgs_len);
  if ((safety_setup.current_safety_mode == SAFETY_ALLOUTPUT) || (safety_setup.current_safety_mode == SAFETY_ELM327)) {
    whitelisted = true;
  }

  bool safety_allowed = false;
  if (whitelisted) {
    safety_allowed = safety_setup.current_hooks->tx(msg);
  }

  return !safety.relay_malfunction && whitelisted && safety_allowed;
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

static int fwd_hook_copy(int bus_num, int addr) {
  bool blocked = safety.relay_malfunction || safety_setup.current_safety_config.disable_forwarding;

  // Block messages that are being checked for relay malfunctions. Safety modes can opt out of this
  // in the case of selective AEB forwarding
  const int destination_bus = get_fwd_bus(bus_num);
  if (!blocked) {
    for (int i = 0; i < safety_setup.current_safety_config.tx_msgs_len; i++) {
      const CanMsg *m = &safety_setup.current_safety_config.tx_msgs[i];
      if (m->check_relay && !m->disable_static_blocking && (m->addr == addr) && (m->bus == (unsigned int)destination_bus)) {
        blocked = true;
        break;
      }
    }
  }

  if (!blocked && (safety_setup.current_hooks->fwd != NULL)) {
    blocked = safety_setup.current_hooks->fwd(bus_num, addr);
  }

  return blocked ? -1 : destination_bus;
}

static void tick_copy(void) {
  safety.safety_mode_cnt += 1U;

  const uint8_t MAX_MISSED_MSGS = 10U;
  bool rx_checks_invalid = false;
  uint32_t ts = safety_ts;
  for (int i=0; i < safety_setup.current_safety_config.rx_checks_len; i++) {
    RxStatus *status = &safety.rx_status[i];
    uint32_t elapsed_time = safety_get_ts_elapsed(ts, status->last_timestamp);
    // lag threshold is max of: 1s and MAX_MISSED_MSGS * expected timestep.
    // Quite conservative to not risk false triggers.
    // 2s of lag is worse case, since the function is called at 1Hz
    uint32_t frequency = safety_setup.current_safety_config.rx_checks[i].msg[status->index].frequency;
    uint32_t timestep = 1000000U / SAFETY_MAX(frequency, 1U);  // frequency < 10Hz is invalid below
    bool lagging = elapsed_time > SAFETY_MAX(timestep * MAX_MISSED_MSGS, 1000000U);
    status->lagging = lagging;

    // enforce minimum frequency for safety-relevant messages
    bool frequency_invalid = frequency < 10U;
    if (lagging || frequency_invalid || !is_msg_valid(i)) {
      rx_checks_invalid = true;
      safety.controls_allowed = false;
    }
  }

  safety.safety_rx_checks_invalid = rx_checks_invalid;
}

static void heartbeat_check_copy(void) {
  // exit controls allowed if unused by openpilot for a few seconds
  if (safety.controls_allowed && !safety.heartbeat_engaged) {
    safety.heartbeat_engaged_mismatches += 1U;
    if (safety.heartbeat_engaged_mismatches >= 3U) {
      safety.controls_allowed = false;
    }
  } else {
    safety.heartbeat_engaged_mismatches = 0U;
  }
}

static void relay_malfunction_set(void) {
  safety.relay_malfunction = true;
}

static void generic_rx_checks(void) {
  safety.gas_pressed_prev = safety.gas_pressed;

  // exit controls on rising edge of brake press
  if (safety.brake_pressed && (!safety.brake_pressed_prev || safety.vehicle_moving)) {
    safety.controls_allowed = false;
  }
  safety.brake_pressed_prev = safety.brake_pressed;

  // exit controls on rising edge of regen paddle
  if (safety.regen_braking && (!safety.regen_braking_prev || safety.vehicle_moving)) {
    safety.controls_allowed = false;
  }
  safety.regen_braking_prev = safety.regen_braking;

  // exit controls on rising edge of steering override/disengage
  if (safety.steering_disengage && !safety.steering_disengage_prev) {
    safety.controls_allowed = false;
  }
  safety.steering_disengage_prev = safety.steering_disengage;
}

static void stock_ecu_check(bool stock_ecu_detected) {
  // allow 1s of transition timeout after relay changes state before assessing malfunctioning
  const uint32_t RELAY_TRNS_TIMEOUT = 1U;

  // check if stock ECU is on bus broken by car harness
  if ((safety.safety_mode_cnt > RELAY_TRNS_TIMEOUT) && stock_ecu_detected) {
    relay_malfunction_set();
  }
}

static int set_hooks_copy(uint16_t mode, uint16_t param) {
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
    {SAFETY_VOLKSWAGEN_MEB, &volkswagen_meb_hooks},
    {SAFETY_NISSAN, &nissan_hooks},
    {SAFETY_NOOUTPUT, &nooutput_hooks},
    {SAFETY_HYUNDAI_LEGACY, &hyundai_legacy_hooks},
    {SAFETY_MAZDA, &mazda_hooks},
    {SAFETY_BODY, &body_hooks},
    {SAFETY_FORD, &ford_hooks},
    {SAFETY_RIVIAN, &rivian_hooks},
    {SAFETY_TESLA, &tesla_hooks},
    {SAFETY_HYUNDAI_CANFD, &hyundai_canfd_hooks},
#ifdef ALLOW_DEBUG
    {SAFETY_MG, &mg_hooks},
    {SAFETY_CHRYSLER_CUSW, &chrysler_cusw_hooks},
    {SAFETY_PSA, &psa_hooks},
    {SAFETY_BYD, &byd_hooks},
    {SAFETY_SUBARU_PREGLOBAL, &subaru_preglobal_hooks},
    {SAFETY_VOLKSWAGEN_MLB, &volkswagen_mlb_hooks},
    {SAFETY_VOLKSWAGEN_PQ, &volkswagen_pq_hooks},
    {SAFETY_ALLOUTPUT, &alloutput_hooks},
#endif
  };

  // reset state set by safety mode
  safety = (SafetyState){
    .heartbeat_engaged = safety.heartbeat_engaged,
    .heartbeat_engaged_mismatches = safety.heartbeat_engaged_mismatches,
  };

  safety_config cfg = {0};

  int set_status = -1;  // not set
  int hook_config_count = sizeof(safety_hook_registry) / sizeof(safety_hook_config);
  for (int i = 0; i < hook_config_count; i++) {
    if (safety_hook_registry[i].id == mode) {
      safety_setup.current_hooks = safety_hook_registry[i].hooks;
      safety_setup.current_safety_mode = mode;
      safety_setup.current_safety_param = param;
      set_status = 0;  // set
    }
  }
  if ((set_status == 0) && (safety_setup.current_hooks->init != NULL)) {
    cfg = safety_setup.current_hooks->init(param);
  }

  // by field, the copies are compared by byte and the padding of cfg is unspecified
  safety_setup.current_safety_config.rx_checks = cfg.rx_checks;
  safety_setup.current_safety_config.rx_checks_len = cfg.rx_checks_len;
  safety_setup.current_safety_config.tx_msgs = cfg.tx_msgs;
  safety_setup.current_safety_config.tx_msgs_len = cfg.tx_msgs_len;
  safety_setup.current_safety_config.disable_forwarding = cfg.disable_forwarding;
  return set_status;
}

// B runs first, so safety is A again once a hook returns
bool safety_rx_hook(const CANPacket_t *msg) {
  safety_ts = microsecond_timer_get();
  current_copy = &safety_b;
  bool valid_b = rx_hook_copy(msg);
  current_copy = &safety_a;
  bool valid = rx_hook_copy(msg);
  (void)safety_redundancy_check(valid == valid_b);
  return valid;
}

bool safety_tx_hook(CANPacket_t *msg) {
  safety_ts = microsecond_timer_get();
  current_copy = &safety_b;
  bool allowed_b = tx_hook_copy(msg);
  current_copy = &safety_a;
  bool allowed = tx_hook_copy(msg);
  (void)safety_redundancy_check(allowed == allowed_b);
  return allowed;
}

int safety_fwd_hook(int bus_num, int addr) {
  current_copy = &safety_b;
  int destination_bus_b = fwd_hook_copy(bus_num, addr);
  current_copy = &safety_a;
  int destination_bus = fwd_hook_copy(bus_num, addr);
  (void)safety_redundancy_check(destination_bus == destination_bus_b);
  return destination_bus;
}

// 1Hz safety function called by main
void safety_tick(void) {
  safety_ts = microsecond_timer_get();
  current_copy = &safety_b;
  tick_copy();
  current_copy = &safety_a;
  tick_copy();
  (void)safety_redundancy_check(safety_states_equal());
}

void safety_heartbeat(bool engaged) {
  safety_a.state.heartbeat_engaged = engaged;
  safety_b.state.heartbeat_engaged = engaged;
}

void safety_alternative_experience(int alt_exp) {
  safety_a.setup.alternative_experience = alt_exp;
  safety_b.setup.alternative_experience = alt_exp;
}

void safety_heartbeat_check(void) {
  current_copy = &safety_b;
  heartbeat_check_copy();
  current_copy = &safety_a;
  heartbeat_check_copy();
}

int set_safety_hooks(uint16_t mode, uint16_t param) {
  current_copy = &safety_b;
  (void)set_hooks_copy(mode, param);
  current_copy = &safety_a;
  return set_hooks_copy(mode, param);
}

// convert a trimmed integer to signed 32 bit int
int to_signed(int d, int bits) {
  int d_signed = d;
  int max_value = (1 << SAFETY_MAX((bits - 1), 0));
  if (d >= max_value) {
    d_signed = d - (1 << SAFETY_MAX(bits, 0));
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
    safety.controls_allowed = false;
  }
  if (cruise_engaged && !safety.cruise_engaged_prev) {
    safety.controls_allowed = true;
  }
  safety.cruise_engaged_prev = cruise_engaged;
}

void speed_mismatch_check(const float speed_2) {
  // Disable controls if speeds from two sources are too far apart.
  // For safety modes that use speed to adjust torque or angle limits
  // compared in vehicle_speed's integer units, so the limit is exact
  const int MAX_SPEED_DELTA = 2 * (int)VEHICLE_SPEED_FACTOR;  // 2 m/s
  bool is_invalid_speed = SAFETY_ABS(ROUND(speed_2 * VEHICLE_SPEED_FACTOR) - safety.vehicle_speed.values[0]) > MAX_SPEED_DELTA;
  if (is_invalid_speed) {
    safety.controls_allowed = false;
  }
}
