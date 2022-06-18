#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#include "panda.h"
#include "can_definitions.h"
#include "utils.h"

#define CANFD

typedef struct {
  uint32_t CNT;
} TIM_TypeDef;

struct sample_t torque_meas;
struct sample_t torque_driver;

TIM_TypeDef timer;
TIM_TypeDef *MICROSECOND_TIMER = &timer;
uint32_t microsecond_timer_get(void);

// from board_declarations.h
#define HW_TYPE_UNKNOWN 0U
#define HW_TYPE_WHITE_PANDA 1U
#define HW_TYPE_GREY_PANDA 2U
#define HW_TYPE_BLACK_PANDA 3U
#define HW_TYPE_PEDAL 4U
#define HW_TYPE_UNO 5U
#define HW_TYPE_DOS 6U

#define ALLOW_DEBUG

// from main_declarations.h
uint8_t hw_type = HW_TYPE_UNKNOWN;

// from config.h
#define MIN(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a < _b ? _a : _b; })

#define MAX(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a > _b ? _a : _b; })

#define ABS(a)                                  \
 ({ __typeof__ (a) _a = (a);                    \
   (_a > 0) ? _a : (-_a); })

// from faults.h
#define FAULT_RELAY_MALFUNCTION         (1U << 0)
void fault_occurred(uint32_t fault) {
}
void fault_recovered(uint32_t fault) {
}

#define UNUSED(x) (void)(x)

#ifndef PANDA
#define PANDA
#endif
#define NULL ((void*)0)
#define static
#include "safety.h"

uint32_t microsecond_timer_get(void) {
  return MICROSECOND_TIMER->CNT;
}

void safety_tick_current_rx_checks() {
  safety_tick(current_rx_checks);
}

bool addr_checks_valid() {
  if (current_rx_checks->len <= 0) {
    printf("missing RX checks\n");
    return false;
  }

  for (int i = 0; i < current_rx_checks->len; i++) {
    const AddrCheckStruct addr = current_rx_checks->check[i];
    bool valid = addr.msg_seen && !addr.lagging && addr.valid_checksum && (addr.wrong_counters < MAX_WRONG_COUNTERS);
    if (!valid) {
      //printf("seen %d lagging %d valid checksum %d wrong counters %d\n", addr.msg_seen, addr.lagging, addr.valid_checksum, addr.wrong_counters);
      return false;
    }
  }
  return true;
}

void set_controls_allowed(bool c){
  controls_allowed = c;
}

void set_alternative_experience(int mode){
  alternative_experience = mode;
}

void set_relay_malfunction(bool c){
  relay_malfunction = c;
}

void set_gas_interceptor_detected(bool c){
  gas_interceptor_detected = c;
}

bool get_controls_allowed(void){
  return controls_allowed;
}

int get_alternative_experience(void){
  return alternative_experience;
}

bool get_relay_malfunction(void){
  return relay_malfunction;
}

bool get_gas_interceptor_detected(void){
  return gas_interceptor_detected;
}

int get_gas_interceptor_prev(void){
  return gas_interceptor_prev;
}

bool get_gas_pressed_prev(void){
  return gas_pressed_prev;
}

bool get_brake_pressed_prev(void){
  return brake_pressed_prev;
}

bool get_cruise_engaged_prev(void){
  return cruise_engaged_prev;
}

bool get_vehicle_moving(void){
  return vehicle_moving;
}

bool get_acc_main_on(void){
  return acc_main_on;
}

int get_hw_type(void){
  return hw_type;
}

void set_timer(uint32_t t){
  timer.CNT = t;
}

void set_torque_meas(int min, int max){
  torque_meas.min = min;
  torque_meas.max = max;
}

int get_torque_meas_min(void){
  return torque_meas.min;
}

int get_torque_meas_max(void){
  return torque_meas.max;
}

void set_torque_driver(int min, int max){
  torque_driver.min = min;
  torque_driver.max = max;
}

int get_torque_driver_min(void){
  return torque_driver.min;
}

int get_torque_driver_max(void){
  return torque_driver.max;
}

void set_rt_torque_last(int t){
  rt_torque_last = t;
}

void set_desired_torque_last(int t){
  desired_torque_last = t;
}

void set_desired_angle_last(int t){
  desired_angle_last = t;
}


// ***** car specific helpers *****

void set_honda_alt_brake_msg(bool c){
  honda_alt_brake_msg = c;
}

void set_honda_bosch_long(bool c){
  honda_bosch_long = c;
}

int get_honda_hw(void) {
  return honda_hw;
}

void set_honda_fwd_brake(bool c){
  honda_fwd_brake = c;
}

void init_tests(void){
  // get HW_TYPE from env variable set in test.sh
  if (getenv("HW_TYPE")) {
    hw_type = atoi(getenv("HW_TYPE"));
  }
  safety_mode_cnt = 2U;  // avoid ignoring relay_malfunction logic
  alternative_experience = 0;
  set_timer(0);
}

void init_tests_honda(void){
  init_tests();
  honda_fwd_brake = false;
}

void set_gmlan_digital_output(int to_set){
}

void reset_gmlan_switch_timeout(void){
}

void gmlan_switch_init(int timeout_enable){
}

