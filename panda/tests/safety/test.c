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

struct sample_t torque_meas;

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

void set_rt_torque_last(int t){
  rt_torque_last = t;
}

void set_desired_torque_last(int t){
  desired_torque_last = t;
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

void init_tests_toyota(void){
  torque_meas.min = 0;
  torque_meas.max = 0;
  desired_torque_last = 0;
  rt_torque_last = 0;
  ts_last = 0;
  set_timer(0);
}

void init_tests_honda(void){
  ego_speed = 0;
  gas_interceptor_detected = 0;
  brake_prev = 0;
  gas_prev = 0;
}
