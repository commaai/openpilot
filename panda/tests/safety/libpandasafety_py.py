import os

from cffi import FFI

can_dir = os.path.dirname(os.path.abspath(__file__))
libpandasafety_fn = os.path.join(can_dir, "libpandasafety.so")

ffi = FFI()
ffi.cdef("""
typedef struct {
  unsigned char reserved : 1;
  unsigned char bus : 3;
  unsigned char data_len_code : 4;
  unsigned char rejected : 1;
  unsigned char returned : 1;
  unsigned char extended : 1;
  unsigned int addr : 29;
  unsigned char data[64];
} CANPacket_t;
""", packed=True)

ffi.cdef("""
typedef struct
{
  uint32_t CNT;
} TIM_TypeDef;

void set_controls_allowed(bool c);
bool get_controls_allowed(void);
bool get_longitudinal_allowed(void);
void set_alternative_experience(int mode);
int get_alternative_experience(void);
void set_relay_malfunction(bool c);
bool get_relay_malfunction(void);
void set_gas_interceptor_detected(bool c);
bool get_gas_interceptor_detected(void);
int get_gas_interceptor_prev(void);
bool get_gas_pressed_prev(void);
bool get_brake_pressed_prev(void);
bool get_acc_main_on(void);

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

int safety_rx_hook(CANPacket_t *to_send);
int safety_tx_hook(CANPacket_t *to_push);
int safety_fwd_hook(int bus_num, CANPacket_t *to_fwd);
int set_safety_hooks(uint16_t mode, uint16_t param);

void safety_tick_current_rx_checks();
bool addr_checks_valid();

void init_tests(void);

void init_tests_honda(void);
void set_honda_fwd_brake(bool);
void set_honda_alt_brake_msg(bool);
void set_honda_bosch_long(bool c);
int get_honda_hw(void);

""")

libpandasafety = ffi.dlopen(libpandasafety_fn)
