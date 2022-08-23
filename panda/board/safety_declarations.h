#define GET_BIT(msg, b) (((msg)->data[((b) / 8U)] >> ((b) % 8U)) & 0x1U)
#define GET_BYTE(msg, b) ((msg)->data[(b)])
#define GET_BYTES_04(msg) ((msg)->data[0] | ((msg)->data[1] << 8U) | ((msg)->data[2] << 16U) | ((msg)->data[3] << 24U))
#define GET_BYTES_48(msg) ((msg)->data[4] | ((msg)->data[5] << 8U) | ((msg)->data[6] << 16U) | ((msg)->data[7] << 24U))
#define GET_FLAG(value, mask) (((__typeof__(mask))(value) & (mask)) == (mask))

const int MAX_WRONG_COUNTERS = 5;
const uint8_t MAX_MISSED_MSGS = 10U;

// sample struct that keeps 6 samples in memory
struct sample_t {
  int values[6];
  int min;
  int max;
} sample_t_default = {.values = {0}, .min = 0, .max = 0};

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

typedef enum {
  TorqueMotorLimited,   // torque steering command, limited by EPS output torque
  TorqueDriverLimited,  // torque steering command, limited by driver's input torque
} SteeringControlType;

typedef struct {
  const int max_steer;
  const int max_rate_up;
  const int max_rate_down;
  const int max_rt_delta;
  const uint32_t max_rt_interval;

  const SteeringControlType type;

  // driver torque limits
  const int driver_torque_allowance;
  const int driver_torque_factor;

  // motor torque limits
  const int max_torque_error;
} SteeringLimits;

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

typedef struct {
  AddrCheckStruct *check;
  int len;
} addr_checks;

int safety_rx_hook(CANPacket_t *to_push);
int safety_tx_hook(CANPacket_t *to_send);
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
bool get_longitudinal_allowed(void);
bool rt_rate_limit_check(int val, int val_last, const int MAX_RT_DELTA);
float interpolate(struct lookup_t xy, float x);
void gen_crc_lookup_table_8(uint8_t poly, uint8_t crc_lut[]);
void gen_crc_lookup_table_16(uint16_t poly, uint16_t crc_lut[]);
bool msg_allowed(CANPacket_t *to_send, const CanMsg msg_list[], int len);
int get_addr_check_index(CANPacket_t *to_push, AddrCheckStruct addr_list[], const int len);
void update_counter(AddrCheckStruct addr_list[], int index, uint8_t counter);
void update_addr_timestamp(AddrCheckStruct addr_list[], int index);
bool is_msg_valid(AddrCheckStruct addr_list[], int index);
bool addr_safety_check(CANPacket_t *to_push,
                       const addr_checks *rx_checks,
                       uint32_t (*get_checksum)(CANPacket_t *to_push),
                       uint32_t (*compute_checksum)(CANPacket_t *to_push),
                       uint8_t (*get_counter)(CANPacket_t *to_push));
void generic_rx_checks(bool stock_ecu_detected);
void relay_malfunction_set(void);
void relay_malfunction_reset(void);
bool steer_torque_cmd_checks(int desired_torque, int steer_req, const SteeringLimits limits);
void pcm_cruise_check(bool cruise_engaged);

typedef const addr_checks* (*safety_hook_init)(uint16_t param);
typedef int (*rx_hook)(CANPacket_t *to_push);
typedef int (*tx_hook)(CANPacket_t *to_send, bool longitudinal_allowed);
typedef int (*tx_lin_hook)(int lin_num, uint8_t *data, int len);
typedef int (*fwd_hook)(int bus_num, CANPacket_t *to_fwd);

typedef struct {
  safety_hook_init init;
  rx_hook rx;
  tx_hook tx;
  tx_lin_hook tx_lin;
  fwd_hook fwd;
} safety_hooks;

void safety_tick(const addr_checks *addr_checks);

// This can be set by the safety hooks
bool controls_allowed = false;
bool relay_malfunction = false;
bool gas_interceptor_detected = false;
int gas_interceptor_prev = 0;
bool gas_pressed = false;
bool gas_pressed_prev = false;
bool brake_pressed = false;
bool brake_pressed_prev = false;
bool cruise_engaged_prev = false;
float vehicle_speed = 0;
bool vehicle_moving = false;
bool acc_main_on = false;  // referred to as "ACC off" in ISO 15622:2018
int cruise_button_prev = 0;

// for safety modes with torque steering control
int desired_torque_last = 0;       // last desired steer torque
int rt_torque_last = 0;            // last desired torque for real time check
struct sample_t torque_meas;       // last 6 motor torques produced by the eps
struct sample_t torque_driver;     // last 6 driver torques measured
uint32_t ts_last = 0;

// state for controls_allowed timeout logic
bool heartbeat_engaged = false;             // openpilot enabled, passed in heartbeat USB command
uint32_t heartbeat_engaged_mismatches = 0;  // count of mismatches between heartbeat_engaged and controls_allowed

// for safety modes with angle steering control
uint32_t ts_angle_last = 0;
int desired_angle_last = 0;
struct sample_t angle_meas;         // last 6 steer angles

// This can be set with a USB command
// It enables features that allow alternative experiences, like not disengaging on gas press
// It is only either 0 or 1 on mainline comma.ai openpilot

#define ALT_EXP_DISABLE_DISENGAGE_ON_GAS 1

// If using this flag, make sure to communicate to your users that a stock safety feature is now disabled.
#define ALT_EXP_DISABLE_STOCK_AEB 2

// If using this flag, be aware that harder braking is more likely to lead to rear endings,
//   and that alone this flag doesn't make braking compliant because there's also a time element.
// Setting this flag is used for allowing the full -5.0 to +4.0 m/s^2 at lower speeds
// See ISO 15622:2018 for more information.
#define ALT_EXP_RAISE_LONGITUDINAL_LIMITS_TO_ISO_MAX 8

int alternative_experience = 0;

// time since safety mode has been changed
uint32_t safety_mode_cnt = 0U;
// allow 1s of transition timeout after relay changes state before assessing malfunctioning
const uint32_t RELAY_TRNS_TIMEOUT = 1U;
