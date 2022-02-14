#include "safety_declarations.h"

// include the safety policies.
#include "safety/safety_defaults.h"
#include "safety/safety_honda.h"
#include "safety/safety_toyota.h"
#include "safety/safety_tesla.h"
#include "safety/safety_gm.h"
#include "safety/safety_ford.h"
#include "safety/safety_hyundai.h"
#include "safety/safety_chrysler.h"
#include "safety/safety_subaru.h"
#include "safety/safety_mazda.h"
#include "safety/safety_nissan.h"
#include "safety/safety_volkswagen_mqb.h"
#include "safety/safety_volkswagen_pq.h"
#include "safety/safety_elm327.h"

// from cereal.car.CarParams.SafetyModel
#define SAFETY_SILENT 0U
#define SAFETY_HONDA_NIDEC 1U
#define SAFETY_TOYOTA 2U
#define SAFETY_ELM327 3U
#define SAFETY_GM 4U
#define SAFETY_HONDA_BOSCH_GIRAFFE 5U
#define SAFETY_FORD 6U
#define SAFETY_HYUNDAI 8U
#define SAFETY_CHRYSLER 9U
#define SAFETY_TESLA 10U
#define SAFETY_SUBARU 11U
#define SAFETY_MAZDA 13U
#define SAFETY_NISSAN 14U
#define SAFETY_VOLKSWAGEN_MQB 15U
#define SAFETY_ALLOUTPUT 17U
#define SAFETY_GM_ASCM 18U
#define SAFETY_NOOUTPUT 19U
#define SAFETY_HONDA_BOSCH 20U
#define SAFETY_VOLKSWAGEN_PQ 21U
#define SAFETY_SUBARU_LEGACY 22U
#define SAFETY_HYUNDAI_LEGACY 23U
#define SAFETY_HYUNDAI_COMMUNITY 24U
#define SAFETY_STELLANTIS 25U

uint16_t current_safety_mode = SAFETY_SILENT;
int16_t current_safety_param = 0;
const safety_hooks *current_hooks = &nooutput_hooks;
const addr_checks *current_rx_checks = &default_rx_checks;

int safety_rx_hook(CANPacket_t *to_push) {
  return current_hooks->rx(to_push);
}

int safety_tx_hook(CANPacket_t *to_send) {
  return (relay_malfunction ? -1 : current_hooks->tx(to_send));
}

int safety_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  return current_hooks->tx_lin(lin_num, data, len);
}

int safety_fwd_hook(int bus_num, CANPacket_t *to_fwd) {
  return (relay_malfunction ? -1 : current_hooks->fwd(bus_num, to_fwd));
}

// Given a CRC-8 poly, generate a static lookup table to use with a fast CRC-8
// algorithm. Called at init time for safety modes using CRC-8.
void gen_crc_lookup_table(uint8_t poly, uint8_t crc_lut[]) {
  for (int i = 0; i < 256; i++) {
    uint8_t crc = i;
    for (int j = 0; j < 8; j++) {
      if ((crc & 0x80U) != 0U)
        crc = (uint8_t)((crc << 1) ^ poly);
      else
        crc <<= 1;
    }
    crc_lut[i] = crc;
  }
}

bool msg_allowed(CANPacket_t *to_send, const CanMsg msg_list[], int len) {
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);
  int length = GET_LEN(to_send);

  bool allowed = false;
  for (int i = 0; i < len; i++) {
    if ((addr == msg_list[i].addr) && (bus == msg_list[i].bus) && (length == msg_list[i].len)) {
      allowed = true;
      break;
    }
  }
  return allowed;
}

// compute the time elapsed (in microseconds) from 2 counter samples
// case where ts < ts_last is ok: overflow is properly re-casted into uint32_t
uint32_t get_ts_elapsed(uint32_t ts, uint32_t ts_last) {
  return ts - ts_last;
}

int get_addr_check_index(CANPacket_t *to_push, AddrCheckStruct addr_list[], const int len) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);
  int length = GET_LEN(to_push);

  int index = -1;
  for (int i = 0; i < len; i++) {
    // if multiple msgs are allowed, determine which one is present on the bus
    if (!addr_list[i].msg_seen) {
      for (uint8_t j = 0U; addr_list[i].msg[j].addr != 0; j++) {
        if ((addr == addr_list[i].msg[j].addr) && (bus == addr_list[i].msg[j].bus) &&
              (length == addr_list[i].msg[j].len)) {
          addr_list[i].index = j;
          addr_list[i].msg_seen = true;
          break;
        }
      }
    }

    int idx = addr_list[i].index;
    if ((addr == addr_list[i].msg[idx].addr) && (bus == addr_list[i].msg[idx].bus) &&
        (length == addr_list[i].msg[idx].len)) {
      index = i;
      break;
    }
  }
  return index;
}

// 1Hz safety function called by main. Now just a check for lagging safety messages
void safety_tick(const addr_checks *rx_checks) {
  uint32_t ts = microsecond_timer_get();
  if (rx_checks != NULL) {
    for (int i=0; i < rx_checks->len; i++) {
      uint32_t elapsed_time = get_ts_elapsed(ts, rx_checks->check[i].last_timestamp);
      // lag threshold is max of: 1s and MAX_MISSED_MSGS * expected timestep.
      // Quite conservative to not risk false triggers.
      // 2s of lag is worse case, since the function is called at 1Hz
      bool lagging = elapsed_time > MAX(rx_checks->check[i].msg[rx_checks->check[i].index].expected_timestep * MAX_MISSED_MSGS, 1e6);
      rx_checks->check[i].lagging = lagging;
      if (lagging) {
        controls_allowed = 0;
      }
    }
  }
}

void update_counter(AddrCheckStruct addr_list[], int index, uint8_t counter) {
  if (index != -1) {
    uint8_t expected_counter = (addr_list[index].last_counter + 1U) % (addr_list[index].msg[addr_list[index].index].max_counter + 1U);
    addr_list[index].wrong_counters += (expected_counter == counter) ? -1 : 1;
    addr_list[index].wrong_counters = MAX(MIN(addr_list[index].wrong_counters, MAX_WRONG_COUNTERS), 0);
    addr_list[index].last_counter = counter;
  }
}

bool is_msg_valid(AddrCheckStruct addr_list[], int index) {
  bool valid = true;
  if (index != -1) {
    if ((!addr_list[index].valid_checksum) || (addr_list[index].wrong_counters >= MAX_WRONG_COUNTERS)) {
      valid = false;
      controls_allowed = 0;
    }
  }
  return valid;
}

void update_addr_timestamp(AddrCheckStruct addr_list[], int index) {
  if (index != -1) {
    uint32_t ts = microsecond_timer_get();
    addr_list[index].last_timestamp = ts;
  }
}

bool addr_safety_check(CANPacket_t *to_push,
                       const addr_checks *rx_checks,
                       uint8_t (*get_checksum)(CANPacket_t *to_push),
                       uint8_t (*compute_checksum)(CANPacket_t *to_push),
                       uint8_t (*get_counter)(CANPacket_t *to_push)) {

  int index = get_addr_check_index(to_push, rx_checks->check, rx_checks->len);
  update_addr_timestamp(rx_checks->check, index);

  if (index != -1) {
    // checksum check
    if ((get_checksum != NULL) && (compute_checksum != NULL) && rx_checks->check[index].msg[rx_checks->check[index].index].check_checksum) {
      uint8_t checksum = get_checksum(to_push);
      uint8_t checksum_comp = compute_checksum(to_push);
      rx_checks->check[index].valid_checksum = checksum_comp == checksum;
    } else {
      rx_checks->check[index].valid_checksum = true;
    }

    // counter check (max_counter == 0 means skip check)
    if ((get_counter != NULL) && (rx_checks->check[index].msg[rx_checks->check[index].index].max_counter > 0U)) {
      uint8_t counter = get_counter(to_push);
      update_counter(rx_checks->check, index, counter);
    } else {
      rx_checks->check[index].wrong_counters = 0U;
    }
  }
  return is_msg_valid(rx_checks->check, index);
}

void generic_rx_checks(bool stock_ecu_detected) {
  // exit controls on rising edge of gas press
  if (gas_pressed && !gas_pressed_prev && !(unsafe_mode & UNSAFE_DISABLE_DISENGAGE_ON_GAS)) {
    controls_allowed = 0;
  }
  gas_pressed_prev = gas_pressed;

  // exit controls on rising edge of brake press
  if (brake_pressed && (!brake_pressed_prev || vehicle_moving)) {
    controls_allowed = 0;
  }
  brake_pressed_prev = brake_pressed;

  // check if stock ECU is on bus broken by car harness
  if ((safety_mode_cnt > RELAY_TRNS_TIMEOUT) && stock_ecu_detected) {
    relay_malfunction_set();
  }
}

void relay_malfunction_set(void) {
  relay_malfunction = true;
  fault_occurred(FAULT_RELAY_MALFUNCTION);
}

void relay_malfunction_reset(void) {
  relay_malfunction = false;
  fault_recovered(FAULT_RELAY_MALFUNCTION);
}

typedef struct {
  uint16_t id;
  const safety_hooks *hooks;
} safety_hook_config;

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
#ifdef ALLOW_DEBUG
  {SAFETY_TESLA, &tesla_hooks},
  {SAFETY_SUBARU_LEGACY, &subaru_legacy_hooks},
  {SAFETY_VOLKSWAGEN_PQ, &volkswagen_pq_hooks},
  {SAFETY_ALLOUTPUT, &alloutput_hooks},
  {SAFETY_FORD, &ford_hooks},
#endif
};

int set_safety_hooks(uint16_t mode, int16_t param) {
  // reset state set by safety mode
  safety_mode_cnt = 0U;
  relay_malfunction = false;
  gas_interceptor_detected = false;
  gas_interceptor_prev = 0;
  gas_pressed = false;
  gas_pressed_prev = false;
  brake_pressed = false;
  brake_pressed_prev = false;
  cruise_engaged_prev = false;
  vehicle_speed = 0;
  vehicle_moving = false;
  acc_main_on = false;
  desired_torque_last = 0;
  rt_torque_last = 0;
  ts_angle_last = 0;
  desired_angle_last = 0;
  ts_last = 0;

  torque_meas.max = 0;
  torque_meas.max = 0;
  torque_driver.min = 0;
  torque_driver.max = 0;
  angle_meas.min = 0;
  angle_meas.max = 0;

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
    current_rx_checks = current_hooks->init(param);
    // reset message index and seen flags in addr struct
    for (int j = 0; j < current_rx_checks->len; j++) {
      current_rx_checks->check[j].index = 0;
      current_rx_checks->check[j].msg_seen = false;
    }
  }
  return set_status;
}

// convert a trimmed integer to signed 32 bit int
int to_signed(int d, int bits) {
  int d_signed = d;
  if (d >= (1 << MAX((bits - 1), 0))) {
    d_signed = d - (1 << MAX(bits, 0));
  }
  return d_signed;
}

// given a new sample, update the smaple_t struct
void update_sample(struct sample_t *sample, int sample_new) {
  int sample_size = sizeof(sample->values) / sizeof(sample->values[0]);
  for (int i = sample_size - 1; i > 0; i--) {
    sample->values[i] = sample->values[i-1];
  }
  sample->values[0] = sample_new;

  // get the minimum and maximum measured samples
  sample->min = sample->values[0];
  sample->max = sample->values[0];
  for (int i = 1; i < sample_size; i++) {
    if (sample->values[i] < sample->min) {
      sample->min = sample->values[i];
    }
    if (sample->values[i] > sample->max) {
      sample->max = sample->values[i];
    }
  }
}

bool max_limit_check(int val, const int MAX_VAL, const int MIN_VAL) {
  return (val > MAX_VAL) || (val < MIN_VAL);
}

// check that commanded value isn't too far from measured
bool dist_to_meas_check(int val, int val_last, struct sample_t *val_meas,
  const int MAX_RATE_UP, const int MAX_RATE_DOWN, const int MAX_ERROR) {

  // *** val rate limit check ***
  int highest_allowed_rl = MAX(val_last, 0) + MAX_RATE_UP;
  int lowest_allowed_rl = MIN(val_last, 0) - MAX_RATE_UP;

  // if we've exceeded the meas val, we must start moving toward 0
  int highest_allowed = MIN(highest_allowed_rl, MAX(val_last - MAX_RATE_DOWN, MAX(val_meas->max, 0) + MAX_ERROR));
  int lowest_allowed = MAX(lowest_allowed_rl, MIN(val_last + MAX_RATE_DOWN, MIN(val_meas->min, 0) - MAX_ERROR));

  // check for violation
  return (val < lowest_allowed) || (val > highest_allowed);
}

// check that commanded value isn't fighting against driver
bool driver_limit_check(int val, int val_last, struct sample_t *val_driver,
  const int MAX_VAL, const int MAX_RATE_UP, const int MAX_RATE_DOWN,
  const int MAX_ALLOWANCE, const int DRIVER_FACTOR) {

  int highest_allowed_rl = MAX(val_last, 0) + MAX_RATE_UP;
  int lowest_allowed_rl = MIN(val_last, 0) - MAX_RATE_UP;

  int driver_max_limit = MAX_VAL + (MAX_ALLOWANCE + val_driver->max) * DRIVER_FACTOR;
  int driver_min_limit = -MAX_VAL + (-MAX_ALLOWANCE + val_driver->min) * DRIVER_FACTOR;

  // if we've exceeded the applied torque, we must start moving toward 0
  int highest_allowed = MIN(highest_allowed_rl, MAX(val_last - MAX_RATE_DOWN,
                                             MAX(driver_max_limit, 0)));
  int lowest_allowed = MAX(lowest_allowed_rl, MIN(val_last + MAX_RATE_DOWN,
                                           MIN(driver_min_limit, 0)));

  // check for violation
  return (val < lowest_allowed) || (val > highest_allowed);
}


// real time check, mainly used for steer torque rate limiter
bool rt_rate_limit_check(int val, int val_last, const int MAX_RT_DELTA) {

  // *** torque real time rate limit check ***
  int highest_val = MAX(val_last, 0) + MAX_RT_DELTA;
  int lowest_val = MIN(val_last, 0) - MAX_RT_DELTA;

  // check for violation
  return (val < lowest_val) || (val > highest_val);
}


// interp function that holds extreme values
float interpolate(struct lookup_t xy, float x) {

  int size = sizeof(xy.x) / sizeof(xy.x[0]);
  float ret = xy.y[size - 1];  // default output is last point

  // x is lower than the first point in the x array. Return the first point
  if (x <= xy.x[0]) {
    ret = xy.y[0];

  } else {
    // find the index such that (xy.x[i] <= x < xy.x[i+1]) and linearly interp
    for (int i=0; i < (size - 1); i++) {
      if (x < xy.x[i+1]) {
        float x0 = xy.x[i];
        float y0 = xy.y[i];
        float dx = xy.x[i+1] - x0;
        float dy = xy.y[i+1] - y0;
        // dx should not be zero as xy.x is supposed to be monotonic
        if (dx <= 0.) {
          dx = 0.0001;
        }
        ret = (dy * (x - x0) / dx) + y0;
        break;
      }
    }
  }
  return ret;
}
