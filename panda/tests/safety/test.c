#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct
{
  uint32_t TIR;  /*!< CAN TX mailbox identifier register */
  uint32_t TDTR; /*!< CAN mailbox data length control and time stamp register */
  uint32_t TDLR; /*!< CAN mailbox data low register */
  uint32_t TDHR; /*!< CAN mailbox data high register */
} CAN_TxMailBox_TypeDef;

typedef struct
{
  uint32_t RIR;  /*!< CAN receive FIFO mailbox identifier register */
  uint32_t RDTR; /*!< CAN receive FIFO mailbox data length control and time stamp register */
  uint32_t RDLR; /*!< CAN receive FIFO mailbox data low register */
  uint32_t RDHR; /*!< CAN receive FIFO mailbox data high register */
} CAN_FIFOMailBox_TypeDef;

typedef struct
{
  uint32_t CNT;
} TIM_TypeDef;

struct sample_t torque_meas;
struct sample_t torque_driver;

TIM_TypeDef timer;
TIM_TypeDef *TIM2 = &timer;

// from board_declarations.h
#define HW_TYPE_UNKNOWN 0U
#define HW_TYPE_WHITE_PANDA 1U
#define HW_TYPE_GREY_PANDA 2U
#define HW_TYPE_BLACK_PANDA 3U
#define HW_TYPE_PEDAL 4U
#define HW_TYPE_UNO 5U

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

// from llcan.h
#define GET_BUS(msg) (((msg)->RDTR >> 4) & 0xFF)
#define GET_LEN(msg) ((msg)->RDTR & 0xf)
#define GET_ADDR(msg) ((((msg)->RIR & 4) != 0) ? ((msg)->RIR >> 3) : ((msg)->RIR >> 21))
#define GET_BYTE(msg, b) (((int)(b) > 3) ? (((msg)->RDHR >> (8U * ((unsigned int)(b) % 4U))) & 0XFFU) : (((msg)->RDLR >> (8U * (unsigned int)(b))) & 0xFFU))
#define GET_BYTES_04(msg) ((msg)->RDLR)
#define GET_BYTES_48(msg) ((msg)->RDHR)

#define UNUSED(x) (void)(x)

#ifndef PANDA
#define PANDA
#endif
#define NULL ((void*)0)
#define static
#include "safety.h"

void set_controls_allowed(bool c){
  controls_allowed = c;
}

void set_unsafe_mode(int mode){
  unsafe_mode = mode;
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

int get_unsafe_mode(void){
  return unsafe_mode;
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

int get_hw_type(void){
  return hw_type;
}

bool get_subaru_global(void){
  return subaru_global;
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

void set_honda_alt_brake_msg(bool c){
  honda_alt_brake_msg = c;
}

int get_honda_hw(void) {
  return honda_hw;
}

void set_honda_fwd_brake(bool c){
  honda_fwd_brake = c;
}

void set_nissan_desired_angle_last(int t){
  nissan_desired_angle_last = t;
}

void init_tests(void){
  // get HW_TYPE from env variable set in test.sh
  hw_type = atoi(getenv("HW_TYPE"));
  safety_mode_cnt = 2U;  // avoid ignoring relay_malfunction logic
  gas_pressed_prev = false;
  brake_pressed_prev = false;
  desired_torque_last = 0;
  rt_torque_last = 0;
  ts_last = 0;
  torque_driver.min = 0;
  torque_driver.max = 0;
  torque_meas.min = 0;
  torque_meas.max = 0;
  vehicle_moving = false;
  unsafe_mode = 0;
  set_timer(0);
}

void init_tests_chrysler(void){
  init_tests();
  chrysler_speed = 0;
}

void init_tests_honda(void){
  init_tests();
  honda_fwd_brake = false;
}

void init_tests_nissan(void){
  init_tests();
  nissan_angle_meas.min = 0;
  nissan_angle_meas.max = 0;
  nissan_desired_angle_last = 0;
  set_timer(0);
}

void set_gmlan_digital_output(int to_set){
}

void reset_gmlan_switch_timeout(void){
}

void gmlan_switch_init(int timeout_enable){
}

