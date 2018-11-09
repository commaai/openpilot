// sample struct that keeps 3 samples in memory
struct sample_t {
  int values[6];
  int min;
  int max;
} sample_t_default = {{0}, 0, 0};

// no float support in STM32F2 micros (cortex-m3)
#ifdef PANDA
struct lookup_t {
  float x[3];
  float y[3];
};
#endif

void safety_rx_hook(CAN_FIFOMailBox_TypeDef *to_push);
int safety_tx_hook(CAN_FIFOMailBox_TypeDef *to_send);
int safety_tx_lin_hook(int lin_num, uint8_t *data, int len);
int safety_ignition_hook();
uint32_t get_ts_elapsed(uint32_t ts, uint32_t ts_last);
int to_signed(int d, int bits);
void update_sample(struct sample_t *sample, int sample_new);
int max_limit_check(int val, const int MAX, const int MIN);
int dist_to_meas_check(int val, int val_last, struct sample_t *val_meas,
  const int MAX_RATE_UP, const int MAX_RATE_DOWN, const int MAX_ERROR);
int driver_limit_check(int val, int val_last, struct sample_t *val_driver,
  const int MAX, const int MAX_RATE_UP, const int MAX_RATE_DOWN,
  const int MAX_ALLOWANCE, const int DRIVER_FACTOR);
int rt_rate_limit_check(int val, int val_last, const int MAX_RT_DELTA);
#ifdef PANDA
float interpolate(struct lookup_t xy, float x);
#endif

typedef void (*safety_hook_init)(int16_t param);
typedef void (*rx_hook)(CAN_FIFOMailBox_TypeDef *to_push);
typedef int (*tx_hook)(CAN_FIFOMailBox_TypeDef *to_send);
typedef int (*tx_lin_hook)(int lin_num, uint8_t *data, int len);
typedef int (*ign_hook)();
typedef int (*fwd_hook)(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd);

typedef struct {
  safety_hook_init init;
  ign_hook ignition;
  rx_hook rx;
  tx_hook tx;
  tx_lin_hook tx_lin;
  fwd_hook fwd;
} safety_hooks;

// This can be set by the safety hooks.
int controls_allowed = 0;

// Include the actual safety policies.
#include "safety/safety_defaults.h"
#include "safety/safety_honda.h"
#include "safety/safety_toyota.h"
#ifdef PANDA
#include "safety/safety_toyota_ipas.h"
#include "safety/safety_tesla.h"
#endif
#include "safety/safety_gm.h"
#include "safety/safety_ford.h"
#include "safety/safety_cadillac.h"
#include "safety/safety_hyundai.h"
#include "safety/safety_elm327.h"

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

#define SAFETY_NOOUTPUT 0
#define SAFETY_HONDA 1
#define SAFETY_TOYOTA 2
#define SAFETY_GM 3
#define SAFETY_HONDA_BOSCH 4
#define SAFETY_FORD 5
#define SAFETY_CADILLAC 6
#define SAFETY_HYUNDAI 7
#define SAFETY_TESLA 8
#define SAFETY_TOYOTA_IPAS 0x1335
#define SAFETY_TOYOTA_NOLIMITS 0x1336
#define SAFETY_ALLOUTPUT 0x1337
#define SAFETY_ELM327 0xE327

const safety_hook_config safety_hook_registry[] = {
  {SAFETY_NOOUTPUT, &nooutput_hooks},
  {SAFETY_HONDA, &honda_hooks},
  {SAFETY_HONDA_BOSCH, &honda_bosch_hooks},
  {SAFETY_TOYOTA, &toyota_hooks},
  {SAFETY_GM, &gm_hooks},
  {SAFETY_FORD, &ford_hooks},
  {SAFETY_CADILLAC, &cadillac_hooks},
  {SAFETY_HYUNDAI, &hyundai_hooks},
  {SAFETY_TOYOTA_NOLIMITS, &toyota_nolimits_hooks},
#ifdef PANDA
  {SAFETY_TOYOTA_IPAS, &toyota_ipas_hooks},
  {SAFETY_TESLA, &tesla_hooks},
#endif
  {SAFETY_ALLOUTPUT, &alloutput_hooks},
  {SAFETY_ELM327, &elm327_hooks},
};

#define HOOK_CONFIG_COUNT (sizeof(safety_hook_registry)/sizeof(safety_hook_config))

int safety_set_mode(uint16_t mode, int16_t param) {
  for (int i = 0; i < HOOK_CONFIG_COUNT; i++) {
    if (safety_hook_registry[i].id == mode) {
      current_hooks = safety_hook_registry[i].hooks;
      if (current_hooks->init) current_hooks->init(param);
      return 0;
    }
  }
  return -1;
}

// compute the time elapsed (in microseconds) from 2 counter samples
uint32_t get_ts_elapsed(uint32_t ts, uint32_t ts_last) {
  return ts > ts_last ? ts - ts_last : (0xFFFFFFFF - ts_last) + 1 + ts;
}

// convert a trimmed integer to signed 32 bit int
int to_signed(int d, int bits) {
  if (d >= (1 << (bits - 1))) {
    d -= (1 << bits);
  }
  return d;
}

// given a new sample, update the smaple_t struct
void update_sample(struct sample_t *sample, int sample_new) {
  for (int i = sizeof(sample->values)/sizeof(sample->values[0]) - 1; i > 0; i--) {
    sample->values[i] = sample->values[i-1];
  }
  sample->values[0] = sample_new;

  // get the minimum and maximum measured samples
  sample->min = sample->max = sample->values[0];
  for (int i = 1; i < sizeof(sample->values)/sizeof(sample->values[0]); i++) {
    if (sample->values[i] < sample->min) sample->min = sample->values[i];
    if (sample->values[i] > sample->max) sample->max = sample->values[i];
  }
}

int max_limit_check(int val, const int MAX, const int MIN) {
  return (val > MAX) || (val < MIN);
}

// check that commanded value isn't too far from measured
int dist_to_meas_check(int val, int val_last, struct sample_t *val_meas,
  const int MAX_RATE_UP, const int MAX_RATE_DOWN, const int MAX_ERROR) {

  // *** val rate limit check ***
  int highest_allowed_val = max(val_last, 0) + MAX_RATE_UP;
  int lowest_allowed_val = min(val_last, 0) - MAX_RATE_UP;

  // if we've exceeded the meas val, we must start moving toward 0
  highest_allowed_val = min(highest_allowed_val, max(val_last - MAX_RATE_DOWN, max(val_meas->max, 0) + MAX_ERROR));
  lowest_allowed_val = max(lowest_allowed_val, min(val_last + MAX_RATE_DOWN, min(val_meas->min, 0) - MAX_ERROR));

  // check for violation
  return (val < lowest_allowed_val) || (val > highest_allowed_val);
}

// check that commanded value isn't fighting against driver
int driver_limit_check(int val, int val_last, struct sample_t *val_driver,
  const int MAX, const int MAX_RATE_UP, const int MAX_RATE_DOWN,
  const int MAX_ALLOWANCE, const int DRIVER_FACTOR) {

  int highest_allowed = max(val_last, 0) + MAX_RATE_UP;
  int lowest_allowed = min(val_last, 0) - MAX_RATE_UP;

  int driver_max_limit = MAX + (MAX_ALLOWANCE + val_driver->max) * DRIVER_FACTOR;
  int driver_min_limit = -MAX + (-MAX_ALLOWANCE + val_driver->min) * DRIVER_FACTOR;

  // if we've exceeded the applied torque, we must start moving toward 0
  highest_allowed = min(highest_allowed, max(val_last - MAX_RATE_DOWN,
                                             max(driver_max_limit, 0)));
  lowest_allowed = max(lowest_allowed, min(val_last + MAX_RATE_DOWN,
                                           min(driver_min_limit, 0)));

  // check for violation
  return (val < lowest_allowed) || (val > highest_allowed);
}


// real time check, mainly used for steer torque rate limiter
int rt_rate_limit_check(int val, int val_last, const int MAX_RT_DELTA) {

  // *** torque real time rate limit check ***
  int highest_val = max(val_last, 0) + MAX_RT_DELTA;
  int lowest_val = min(val_last, 0) - MAX_RT_DELTA;

  // check for violation
  return (val < lowest_val) || (val > highest_val);
}


#ifdef PANDA
// interp function that holds extreme values
float interpolate(struct lookup_t xy, float x) {
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
#endif
