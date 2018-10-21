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

TIM_TypeDef timer;
TIM_TypeDef *TIM2 = &timer;

#define min(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a < _b ? _a : _b; })

#define max(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a > _b ? _a : _b; })


#define PANDA
#define static
#include "safety.h"

void set_controls_allowed(int c){
  controls_allowed = c;
}

void reset_angle_control(void){
  angle_control = 0;
}

int get_controls_allowed(void){
  return controls_allowed;
}

void set_timer(int t){
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

int get_ego_speed(void){
  return ego_speed;
}

int get_brake_prev(void){
  return brake_prev;
}

int get_gas_prev(void){
  return gas_prev;
}

void set_honda_alt_brake_msg(bool c){
  honda_alt_brake_msg = c;
}

void set_bosch_hardware(bool c){
  bosch_hardware = c;
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

void init_tests_honda(void){
  ego_speed = 0;
  gas_interceptor_detected = 0;
  brake_prev = 0;
  gas_prev = 0;
}


void set_gmlan_digital_output(int to_set){
}

void reset_gmlan_switch_timeout(void){
}

void gmlan_switch_init(int timeout_enable){
}
