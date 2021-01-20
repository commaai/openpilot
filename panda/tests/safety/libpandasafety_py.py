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
void set_unsafe_mode(int mode);
int get_unsafe_mode(void);
void set_relay_malfunction(bool c);
bool get_relay_malfunction(void);
void set_gas_interceptor_detected(bool c);
bool get_gas_interceptor_detetcted(void);
int get_gas_interceptor_prev(void);
bool get_gas_pressed_prev(void);
bool get_brake_pressed_prev(void);

void set_torque_meas(int min, int max);
int get_torque_meas_min(void);
int get_torque_meas_max(void);
void set_torque_driver(int min, int max);
int get_torque_driver_min(void);
int get_torque_driver_max(void);
void set_desired_torque_last(int t);
void set_rt_torque_last(int t);
void set_desired_angle_last(int t);

bool get_cruise_engaged_prev(void);
bool get_vehicle_moving(void);
int get_hw_type(void);
void set_timer(uint32_t t);

int safety_rx_hook(CAN_FIFOMailBox_TypeDef *to_send);
int safety_tx_hook(CAN_FIFOMailBox_TypeDef *to_push);
int safety_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd);
int set_safety_hooks(uint16_t  mode, int16_t param);

void init_tests(void);

void init_tests_honda(void);
void set_honda_fwd_brake(bool);
void set_honda_alt_brake_msg(bool);
void set_honda_bosch_long(bool c);
int get_honda_hw(void);

""")

libpandasafety = ffi.dlopen(libpandasafety_fn)
