const int MAX_WRONG_COUNTERS = 5;
const uint8_t MAX_MISSED_MSGS = 10U;

// sample struct that keeps 3 samples in memory
struct sample_t {
  int values[6];
  int min;
  int max;
} sample_t_default = {{0}, 0, 0};

// safety code requires floats
struct lookup_t {
  float x[3];
  float y[3];
};

typedef struct {
  int addr;
  int bus;
  int len;
} CanMsg;

typedef struct {
  const int addr;
  const int bus;
  const int len;
  const bool check_checksum;         // true is checksum check is performed
  const uint8_t max_counter;         // maximum value of the counter. 0 means that the counter check is skipped
  const uint32_t expected_timestep;  // expected time between message updates [us]
} CanMsgCheck;

// params and flags about checksum, counter and frequency checks for each monitored address
typedef struct {
  // const params
  const CanMsgCheck msg[3];          // check either messages (e.g. honda steer). Array MUST terminate with an empty struct to know its length.
  // dynamic flags
  bool msg_seen;
  int index;                         // if multiple messages are allowed to be checked, this stores the index of the first one seen. only msg[msg_index] will be used
  bool valid_checksum;               // true if and only if checksum check is passed
  int wrong_counters;                // counter of wrong counters, saturated between 0 and MAX_WRONG_COUNTERS
  uint8_t last_counter;              // last counter value
  uint32_t last_timestamp;           // micro-s
  bool lagging;                      // true if and only if the time between updates is excessive
} AddrCheckStruct;

int safety_rx_hook(CAN_FIFOMailBox_TypeDef *to_push);
int safety_tx_hook(CAN_FIFOMailBox_TypeDef *to_send);
int safety_tx_lin_hook(int lin_num, uint8_t *data, int len);
uint32_t get_ts_elapsed(uint32_t ts, uint32_t ts_last);
int to_signed(int d, int bits);
void update_sample(struct sample_t *sample, int sample_new);
bool max_limit_check(int val, const int MAX, const int MIN);
bool dist_to_meas_check(int val, int val_last, struct sample_t *val_meas,
  const int MAX_RATE_UP, const int MAX_RATE_DOWN, const int MAX_ERROR);
bool driver_limit_check(int val, int val_last, struct sample_t *val_driver,
  const int MAX, const int MAX_RATE_UP, const int MAX_RATE_DOWN,
  const int MAX_ALLOWANCE, const int DRIVER_FACTOR);
bool rt_rate_limit_check(int val, int val_last, const int MAX_RT_DELTA);
float interpolate(struct lookup_t xy, float x);
void gen_crc_lookup_table(uint8_t poly, uint8_t crc_lut[]);
bool msg_allowed(CAN_FIFOMailBox_TypeDef *to_send, const CanMsg msg_list[], int len);
int get_addr_check_index(CAN_FIFOMailBox_TypeDef *to_push, AddrCheckStruct addr_list[], const int len);
void update_counter(AddrCheckStruct addr_list[], int index, uint8_t counter);
void update_addr_timestamp(AddrCheckStruct addr_list[], int index);
bool is_msg_valid(AddrCheckStruct addr_list[], int index);
bool addr_safety_check(CAN_FIFOMailBox_TypeDef *to_push,
                       AddrCheckStruct *addr_check,
                       const int addr_check_len,
                       uint8_t (*get_checksum)(CAN_FIFOMailBox_TypeDef *to_push),
                       uint8_t (*compute_checksum)(CAN_FIFOMailBox_TypeDef *to_push),
                       uint8_t (*get_counter)(CAN_FIFOMailBox_TypeDef *to_push));
void relay_malfunction_set(void);
void relay_malfunction_reset(void);

typedef void (*safety_hook_init)(int16_t param);
typedef int (*rx_hook)(CAN_FIFOMailBox_TypeDef *to_push);
typedef int (*tx_hook)(CAN_FIFOMailBox_TypeDef *to_send);
typedef int (*tx_lin_hook)(int lin_num, uint8_t *data, int len);
typedef int (*fwd_hook)(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd);

typedef struct {
  safety_hook_init init;
  rx_hook rx;
  tx_hook tx;
  tx_lin_hook tx_lin;
  fwd_hook fwd;
  AddrCheckStruct *addr_check;
  const int addr_check_len;
} safety_hooks;

void safety_tick(const safety_hooks *hooks);

// This can be set by the safety hooks
bool controls_allowed = false;
bool relay_malfunction = false;
bool gas_interceptor_detected = false;
int gas_interceptor_prev = 0;
bool gas_pressed_prev = false;
bool brake_pressed_prev = false;
bool cruise_engaged_prev = false;
float vehicle_speed = 0;
bool vehicle_moving = false;

// for safety modes with torque steering control
int desired_torque_last = 0;       // last desired steer torque
int rt_torque_last = 0;            // last desired torque for real time check
struct sample_t torque_meas;       // last 3 motor torques produced by the eps
struct sample_t torque_driver;     // last 3 driver torques measured
uint32_t ts_last = 0;

// for safety modes with angle steering control
uint32_t ts_angle_last = 0;
int desired_angle_last = 0;
struct sample_t angle_meas;         // last 3 steer angles

// This can be set with a USB command
// It enables features we consider to be unsafe, but understand others may have different opinions
// It is always 0 on mainline comma.ai openpilot

// If using this flag, be very careful about what happens if your fork wants to brake while the
//   user is pressing the gas. Tesla is careful with this.
#define UNSAFE_DISABLE_DISENGAGE_ON_GAS 1

// If using this flag, make sure to communicate to your users that a stock safety feature is now disabled.
#define UNSAFE_DISABLE_STOCK_AEB 2

// If using this flag, be aware that harder braking is more likely to lead to rear endings,
//   and that alone this flag doesn't make braking compliant because there's also a time element.
// See ISO 15622:2018 for more information.
#define UNSAFE_RAISE_LONGITUDINAL_LIMITS_TO_ISO_MAX 8

int unsafe_mode = 0;

// time since safety mode has been changed
uint32_t safety_mode_cnt = 0U;
// allow 1s of transition timeout after relay changes state before assessing malfunctioning
const uint32_t RELAY_TRNS_TIMEOUT = 1U;
