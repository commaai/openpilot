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

struct sample_t toyota_torque_meas;
struct sample_t cadillac_torque_driver;
struct sample_t gm_torque_driver;
struct sample_t hyundai_torque_driver;
struct sample_t chrysler_torque_meas;
struct sample_t subaru_torque_driver;
struct sample_t volkswagen_torque_driver;

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

// from board.h
bool board_has_relay(void) {
  return hw_type == HW_TYPE_BLACK_PANDA || hw_type == HW_TYPE_UNO;
}

// from config.h
#define MIN(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a < _b ? _a : _b; })

#define MAX(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a > _b ? _a : _b; })

// from llcan.h
#define GET_BUS(msg) (((msg)->RDTR >> 4) & 0xFF)
#define GET_LEN(msg) ((msg)->RDTR & 0xf)
#define GET_ADDR(msg) ((((msg)->RIR & 4) != 0) ? ((msg)->RIR >> 3) : ((msg)->RIR >> 21))
#define GET_BYTE(msg, b) (((int)(b) > 3) ? (((msg)->RDHR >> (8U * ((unsigned int)(b) % 4U))) & 0XFFU) : (((msg)->RDLR >> (8U * (unsigned int)(b))) & 0xFFU))
#define GET_BYTES_04(msg) ((msg)->RDLR)
#define GET_BYTES_48(msg) ((msg)->RDHR)

#define UNUSED(x) (void)(x)

#define PANDA
#define NULL ((void*)0)
#define static
#include "safety.h"

void set_controls_allowed(bool c){
  controls_allowed = c;
}

void set_relay_malfunction(bool c){
  relay_malfunction = c;
}

void set_long_controls_allowed(bool c){
  long_controls_allowed = c;
}

void set_gas_interceptor_detected(bool c){
  gas_interceptor_detected = c;
}

void reset_angle_control(void){
  angle_control = 0;
}

bool get_controls_allowed(void){
  return controls_allowed;
}

bool get_relay_malfunction(void){
  return relay_malfunction;
}

bool get_long_controls_allowed(void){
  return long_controls_allowed;
}

bool get_gas_interceptor_detected(void){
  return gas_interceptor_detected;
}

int get_gas_interceptor_prev(void){
  return gas_interceptor_prev;
}

int get_hw_type(void){
  return hw_type;
}

void set_timer(uint32_t t){
  timer.CNT = t;
}

void set_toyota_torque_meas(int min, int max){
  toyota_torque_meas.min = min;
  toyota_torque_meas.max = max;
}

void set_cadillac_torque_driver(int min, int max){
  cadillac_torque_driver.min = min;
  cadillac_torque_driver.max = max;
}

void set_gm_torque_driver(int min, int max){
  gm_torque_driver.min = min;
  gm_torque_driver.max = max;
}

void set_hyundai_torque_driver(int min, int max){
  hyundai_torque_driver.min = min;
  hyundai_torque_driver.max = max;
}

void set_chrysler_torque_meas(int min, int max){
  chrysler_torque_meas.min = min;
  chrysler_torque_meas.max = max;
}

void set_subaru_torque_driver(int min, int max){
  subaru_torque_driver.min = min;
  subaru_torque_driver.max = max;
}

void set_volkswagen_torque_driver(int min, int max){
  volkswagen_torque_driver.min = min;
  volkswagen_torque_driver.max = max;
}

int get_chrysler_torque_meas_min(void){
  return chrysler_torque_meas.min;
}

int get_chrysler_torque_meas_max(void){
  return chrysler_torque_meas.max;
}

int get_toyota_gas_prev(void){
  return toyota_gas_prev;
}

int get_toyota_torque_meas_min(void){
  return toyota_torque_meas.min;
}

int get_toyota_torque_meas_max(void){
  return toyota_torque_meas.max;
}

void set_toyota_rt_torque_last(int t){
  toyota_rt_torque_last = t;
}

void set_cadillac_rt_torque_last(int t){
  cadillac_rt_torque_last = t;
}

void set_gm_rt_torque_last(int t){
  gm_rt_torque_last = t;
}

void set_hyundai_rt_torque_last(int t){
  hyundai_rt_torque_last = t;
}

void set_chrysler_rt_torque_last(int t){
  chrysler_rt_torque_last = t;
}

void set_subaru_rt_torque_last(int t){
  subaru_rt_torque_last = t;
}

void set_volkswagen_rt_torque_last(int t){
  volkswagen_rt_torque_last = t;
}

void set_toyota_desired_torque_last(int t){
  toyota_desired_torque_last = t;
}

void set_cadillac_desired_torque_last(int t){
  for (int i = 0; i < 4; i++) cadillac_desired_torque_last[i] = t;
}

void set_gm_desired_torque_last(int t){
  gm_desired_torque_last = t;
}

void set_hyundai_desired_torque_last(int t){
  hyundai_desired_torque_last = t;
}

void set_chrysler_desired_torque_last(int t){
  chrysler_desired_torque_last = t;
}

void set_subaru_desired_torque_last(int t){
  subaru_desired_torque_last = t;
}

void set_volkswagen_desired_torque_last(int t){
  volkswagen_desired_torque_last = t;
}

int get_volkswagen_gas_prev(void){
  return volkswagen_gas_prev;
}

bool get_honda_moving(void){
  return honda_moving;
}

bool get_honda_brake_pressed_prev(void){
  return honda_brake_pressed_prev;
}

int get_honda_gas_prev(void){
  return honda_gas_prev;
}

void set_honda_alt_brake_msg(bool c){
  honda_alt_brake_msg = c;
}

void set_honda_bosch_hardware(bool c){
  honda_bosch_hardware = c;
}

int get_honda_bosch_hardware(void) {
  return honda_bosch_hardware;
}

void set_honda_fwd_brake(bool c){
  honda_fwd_brake = c;
}

void init_tests(void){
  // get HW_TYPE from env variable set in test.sh
  hw_type = atoi(getenv("HW_TYPE"));
  safety_mode_cnt = 2U;  // avoid ignoring relay_malfunction logic
}

void init_tests_toyota(void){
  init_tests();
  toyota_torque_meas.min = 0;
  toyota_torque_meas.max = 0;
  toyota_desired_torque_last = 0;
  toyota_rt_torque_last = 0;
  toyota_ts_last = 0;
  set_timer(0);
}

void init_tests_cadillac(void){
  init_tests();
  cadillac_torque_driver.min = 0;
  cadillac_torque_driver.max = 0;
  for (int i = 0; i < 4; i++) cadillac_desired_torque_last[i] = 0;
  cadillac_rt_torque_last = 0;
  cadillac_ts_last = 0;
  set_timer(0);
}

void init_tests_gm(void){
  init_tests();
  gm_torque_driver.min = 0;
  gm_torque_driver.max = 0;
  gm_desired_torque_last = 0;
  gm_rt_torque_last = 0;
  gm_ts_last = 0;
  set_timer(0);
}

void init_tests_hyundai(void){
  init_tests();
  hyundai_torque_driver.min = 0;
  hyundai_torque_driver.max = 0;
  hyundai_desired_torque_last = 0;
  hyundai_rt_torque_last = 0;
  hyundai_ts_last = 0;
  set_timer(0);
}

void init_tests_chrysler(void){
  init_tests();
  chrysler_torque_meas.min = 0;
  chrysler_torque_meas.max = 0;
  chrysler_desired_torque_last = 0;
  chrysler_rt_torque_last = 0;
  chrysler_ts_last = 0;
  set_timer(0);
}

void init_tests_subaru(void){
  init_tests();
  subaru_torque_driver.min = 0;
  subaru_torque_driver.max = 0;
  subaru_desired_torque_last = 0;
  subaru_rt_torque_last = 0;
  subaru_ts_last = 0;
  set_timer(0);
}

void init_tests_volkswagen(void){
  init_tests();
  volkswagen_torque_driver.min = 0;
  volkswagen_torque_driver.max = 0;
  volkswagen_desired_torque_last = 0;
  volkswagen_rt_torque_last = 0;
  volkswagen_ts_last = 0;
  set_timer(0);
}

void init_tests_honda(void){
  init_tests();
  honda_moving = false;
  honda_brake_pressed_prev = false;
  honda_gas_prev = 0;
  honda_fwd_brake = false;
}

void set_gmlan_digital_output(int to_set){
}

void reset_gmlan_switch_timeout(void){
}

void gmlan_switch_init(int timeout_enable){
}

