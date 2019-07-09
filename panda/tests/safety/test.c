#include <stdint.h>
#include <stdbool.h>

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

TIM_TypeDef timer;
TIM_TypeDef *TIM2 = &timer;

#define MIN(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a < _b ? _a : _b; })

#define MAX(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a > _b ? _a : _b; })

#define UNUSED(x) (void)(x)

#define PANDA
#define NULL ((void*)0)
#define static
#include "safety.h"

void set_controls_allowed(bool c){
  controls_allowed = c;
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

bool get_long_controls_allowed(void){
  return long_controls_allowed;
}

bool get_gas_interceptor_detected(void){
  return gas_interceptor_detected;
}

int get_gas_interceptor_prev(void){
  return gas_interceptor_prev;
}

void set_timer(uint32_t t){
  timer.CNT = t;
}

void set_toyota_camera_forwarded(int t){
  toyota_camera_forwarded = t;
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

void set_hyundai_camera_bus(int t){
  hyundai_camera_bus = t;
}

void set_hyundai_giraffe_switch_2(int t){
  hyundai_giraffe_switch_2 = t;
}

void set_chrysler_camera_detected(int t){
  chrysler_camera_detected = t;
}

void set_chrysler_torque_meas(int min, int max){
  chrysler_torque_meas.min = min;
  chrysler_torque_meas.max = max;
}

void set_subaru_torque_driver(int min, int max){
  subaru_torque_driver.min = min;
  subaru_torque_driver.max = max;
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

int get_honda_ego_speed(void){
  return honda_ego_speed;
}

int get_honda_brake_prev(void){
  return honda_brake_prev;
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

void init_tests_toyota(void){
  toyota_torque_meas.min = 0;
  toyota_torque_meas.max = 0;
  toyota_desired_torque_last = 0;
  toyota_rt_torque_last = 0;
  toyota_ts_last = 0;
  set_timer(0);
}

void init_tests_cadillac(void){
  cadillac_torque_driver.min = 0;
  cadillac_torque_driver.max = 0;
  for (int i = 0; i < 4; i++) cadillac_desired_torque_last[i] = 0;
  cadillac_rt_torque_last = 0;
  cadillac_ts_last = 0;
  set_timer(0);
}

void init_tests_gm(void){
  gm_torque_driver.min = 0;
  gm_torque_driver.max = 0;
  gm_desired_torque_last = 0;
  gm_rt_torque_last = 0;
  gm_ts_last = 0;
  set_timer(0);
}

void init_tests_hyundai(void){
  hyundai_torque_driver.min = 0;
  hyundai_torque_driver.max = 0;
  hyundai_desired_torque_last = 0;
  hyundai_rt_torque_last = 0;
  hyundai_ts_last = 0;
  set_timer(0);
}

void init_tests_chrysler(void){
  chrysler_torque_meas.min = 0;
  chrysler_torque_meas.max = 0;
  chrysler_desired_torque_last = 0;
  chrysler_rt_torque_last = 0;
  chrysler_ts_last = 0;
  set_timer(0);
}

void init_tests_subaru(void){
  subaru_torque_driver.min = 0;
  subaru_torque_driver.max = 0;
  subaru_desired_torque_last = 0;
  subaru_rt_torque_last = 0;
  subaru_ts_last = 0;
  set_timer(0);
}

void init_tests_honda(void){
  honda_ego_speed = 0;
  honda_brake_prev = 0;
  honda_gas_prev = 0;
}

void set_gmlan_digital_output(int to_set){
}

void reset_gmlan_switch_timeout(void){
}

void gmlan_switch_init(int timeout_enable){
}

