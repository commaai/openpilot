import os
import subprocess

from cffi import FFI

can_dir = os.path.dirname(os.path.abspath(__file__))
libpandasafety_fn = os.path.join(can_dir, "libpandasafety.so")
subprocess.check_call(["make"], cwd=can_dir)

ffi = FFI()
ffi.cdef("""
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

void set_controls_allowed(bool c);
bool get_controls_allowed(void);
void set_long_controls_allowed(bool c);
bool get_long_controls_allowed(void);
void set_gas_interceptor_detected(bool c);
bool get_gas_interceptor_detetcted(void);
int get_gas_interceptor_prev(void);
void set_timer(uint32_t t);
void reset_angle_control(void);

void safety_rx_hook(CAN_FIFOMailBox_TypeDef *to_send);
int safety_tx_hook(CAN_FIFOMailBox_TypeDef *to_push);
int safety_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd);
int safety_set_mode(uint16_t  mode, int16_t param);

void init_tests_toyota(void);
int get_toyota_torque_meas_min(void);
int get_toyota_torque_meas_max(void);
int get_toyota_gas_prev(void);
void set_toyota_torque_meas(int min, int max);
void set_toyota_desired_torque_last(int t);
void set_toyota_camera_forwarded(int t);
void set_toyota_rt_torque_last(int t);

void init_tests_honda(void);
int get_honda_ego_speed(void);
int get_honda_brake_prev(void);
int get_honda_gas_prev(void);
void set_honda_alt_brake_msg(bool);
void set_honda_bosch_hardware(bool);

void init_tests_cadillac(void);
void set_cadillac_desired_torque_last(int t);
void set_cadillac_rt_torque_last(int t);
void set_cadillac_torque_driver(int min, int max);

void init_tests_gm(void);
void set_gm_desired_torque_last(int t);
void set_gm_rt_torque_last(int t);
void set_gm_torque_driver(int min, int max);

void init_tests_hyundai(void);
void set_hyundai_desired_torque_last(int t);
void set_hyundai_rt_torque_last(int t);
void set_hyundai_torque_driver(int min, int max);
void set_hyundai_giraffe_switch_2(int t);
void set_hyundai_camera_bus(int t);

void init_tests_chrysler(void);
void set_chrysler_desired_torque_last(int t);
void set_chrysler_rt_torque_last(int t);
void set_chrysler_camera_detected(int t);
int get_chrysler_torque_meas_min(void);
int get_chrysler_torque_meas_max(void);
void set_chrysler_torque_meas(int min, int max);

void init_tests_subaru(void);
void set_subaru_desired_torque_last(int t);
void set_subaru_rt_torque_last(int t);
void set_subaru_torque_driver(int min, int max);


""")

libpandasafety = ffi.dlopen(libpandasafety_fn)
