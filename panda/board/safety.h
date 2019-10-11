// include first, needed by safety policies
#include "safety_declarations.h"
// Include the actual safety policies.
#include "safety/safety_defaults.h"
#include "safety/safety_honda.h"
#include "safety/safety_toyota.h"
#include "safety/safety_toyota_ipas.h"
#include "safety/safety_tesla.h"
#include "safety/safety_gm_ascm.h"
#include "safety/safety_gm.h"
#include "safety/safety_ford.h"
#include "safety/safety_cadillac.h"
#include "safety/safety_hyundai.h"
#include "safety/safety_chrysler.h"
#include "safety/safety_subaru.h"
#include "safety/safety_mazda.h"
#include "safety/safety_elm327.h"

// from cereal.car.CarParams.SafetyModel
#define SAFETY_NOOUTPUT 0U
#define SAFETY_HONDA 1U
#define SAFETY_TOYOTA 2U
#define SAFETY_ELM327 3U
#define SAFETY_GM 4U
#define SAFETY_HONDA_BOSCH 5U
#define SAFETY_FORD 6U
#define SAFETY_CADILLAC 7U
#define SAFETY_HYUNDAI 8U
#define SAFETY_CHRYSLER 9U
#define SAFETY_TESLA 10U
#define SAFETY_SUBARU 11U
#define SAFETY_GM_PASSIVE 12U
#define SAFETY_MAZDA 13U
#define SAFETY_TOYOTA_IPAS 16U
#define SAFETY_ALLOUTPUT 17U
#define SAFETY_GM_ASCM 18U

uint16_t current_safety_mode = SAFETY_NOOUTPUT;
const safety_hooks *current_hooks = &nooutput_hooks;

void safety_rx_hook(CAN_FIFOMailBox_TypeDef *to_push){
  current_hooks->rx(to_push);
}

int safety_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  return current_hooks->tx(to_send);
}

int safety_tx_lin_hook(int lin_num, uint8_t *data, int len){
  return current_hooks->tx_lin(lin_num, data, len);
}

// -1 = Disabled (Use GPIO to determine ignition)
// 0 = Off (not started)
// 1 = On (started)
int safety_ignition_hook() {
  return current_hooks->ignition();
}
int safety_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  return current_hooks->fwd(bus_num, to_fwd);
}

typedef struct {
  uint16_t id;
  const safety_hooks *hooks;
} safety_hook_config;

const safety_hook_config safety_hook_registry[] = {
  {SAFETY_NOOUTPUT, &nooutput_hooks},
  {SAFETY_HONDA, &honda_hooks},
  {SAFETY_TOYOTA, &toyota_hooks},
  {SAFETY_ELM327, &elm327_hooks},
  {SAFETY_GM, &gm_hooks},
  {SAFETY_HONDA_BOSCH, &honda_bosch_hooks},
  {SAFETY_FORD, &ford_hooks},
  {SAFETY_CADILLAC, &cadillac_hooks},
  {SAFETY_HYUNDAI, &hyundai_hooks},
  {SAFETY_CHRYSLER, &chrysler_hooks},
  {SAFETY_TESLA, &tesla_hooks},
  {SAFETY_SUBARU, &subaru_hooks},
  {SAFETY_GM_PASSIVE, &gm_passive_hooks},
  {SAFETY_MAZDA, &mazda_hooks},
  {SAFETY_TOYOTA_IPAS, &toyota_ipas_hooks},
  {SAFETY_ALLOUTPUT, &alloutput_hooks},
  {SAFETY_GM_ASCM, &gm_ascm_hooks},
};

int safety_set_mode(uint16_t mode, int16_t param) {
  int set_status = -1;   // not set
  int hook_config_count = sizeof(safety_hook_registry) / sizeof(safety_hook_config);
  for (int i = 0; i < hook_config_count; i++) {
    if (safety_hook_registry[i].id == mode) {
      current_hooks = safety_hook_registry[i].hooks;
      current_safety_mode = safety_hook_registry[i].id;
      set_status = 0;    // set
      break;
    }
  }
  if ((set_status == 0) && (current_hooks->init != NULL)) {
    current_hooks->init(param);
  }
  return set_status;
}

// compute the time elapsed (in microseconds) from 2 counter samples
// case where ts < ts_last is ok: overflow is properly re-casted into uint32_t
uint32_t get_ts_elapsed(uint32_t ts, uint32_t ts_last) {
  return ts - ts_last;
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
        // dx should not be zero as xy.x is supposed ot be monotonic
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
