#pragma once

#include <stdint.h>
#include <stdbool.h>

#define GET_BIT(msg, b) ((bool)!!(((msg)->data[((b) / 8U)] >> ((b) % 8U)) & 0x1U))
#define GET_BYTE(msg, b) ((msg)->data[(b)])
#define GET_FLAG(value, mask) (((__typeof__(mask))(value) & (mask)) == (mask)) // cppcheck-suppress misra-c2012-1.2; allow __typeof__

#define BUILD_SAFETY_CFG(rx, tx) ((safety_config){(rx), (sizeof((rx)) / sizeof((rx)[0])), \
                                                  (tx), (sizeof((tx)) / sizeof((tx)[0]))})
#define SET_RX_CHECKS(rx, config) \
  do { \
    (config).rx_checks = (rx); \
    (config).rx_checks_len = sizeof((rx)) / sizeof((rx)[0]); \
  } while (0);

#define SET_TX_MSGS(tx, config) \
  do { \
    (config).tx_msgs = (tx); \
    (config).tx_msgs_len = sizeof((tx)) / sizeof((tx)[0]); \
  } while(0);

#define UPDATE_VEHICLE_SPEED(val_ms) (update_sample(&vehicle_speed, ROUND((val_ms) * VEHICLE_SPEED_FACTOR)))

uint32_t GET_BYTES(const CANPacket_t *msg, int start, int len);

extern const int MAX_WRONG_COUNTERS;
#define MAX_ADDR_CHECK_MSGS 3U
#define MAX_SAMPLE_VALS 6
// used to represent floating point vehicle speed in a sample_t
#define VEHICLE_SPEED_FACTOR 100.0


// sample struct that keeps 6 samples in memory
struct sample_t {
  int values[MAX_SAMPLE_VALS];
  int min;
  int max;
};

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
  // torque cmd limits
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

  // safety around steer req bit
  const int min_valid_request_frames;
  const int max_invalid_request_frames;
  const uint32_t min_valid_request_rt_interval;
  const bool has_steer_req_tolerance;

  // angle cmd limits
  const float angle_deg_to_can;
  const struct lookup_t angle_rate_up_lookup;
  const struct lookup_t angle_rate_down_lookup;
  const int max_angle_error;             // used to limit error between meas and cmd while enabled
  const float angle_error_min_speed;     // minimum speed to start limiting angle error

  const bool enforce_angle_error;        // enables max_angle_error check
  const bool inactive_angle_is_zero;     // if false, enforces angle near meas when disabled (default)
} SteeringLimits;

typedef struct {
  // acceleration cmd limits
  const int max_accel;
  const int min_accel;
  const int inactive_accel;

  // gas & brake cmd limits
  // inactive and min gas are 0 on most safety modes
  const int max_gas;
  const int min_gas;
  const int inactive_gas;
  const int max_brake;

  // transmission rpm limits
  const int max_transmission_rpm;
  const int min_transmission_rpm;
  const int inactive_transmission_rpm;

  // speed cmd limits
  const int inactive_speed;
} LongitudinalLimits;

typedef struct {
  const int addr;
  const int bus;
  const int len;
  const bool check_checksum;         // true is checksum check is performed
  const uint8_t max_counter;         // maximum value of the counter. 0 means that the counter check is skipped
  const bool quality_flag;           // true is quality flag check is performed
  const uint32_t frequency;      // expected frequency of the message [Hz]
} CanMsgCheck;

typedef struct {
  // dynamic flags, reset on safety mode init
  bool msg_seen;
  int index;                         // if multiple messages are allowed to be checked, this stores the index of the first one seen. only msg[msg_index] will be used
  bool valid_checksum;               // true if and only if checksum check is passed
  int wrong_counters;                // counter of wrong counters, saturated between 0 and MAX_WRONG_COUNTERS
  bool valid_quality_flag;           // true if the message's quality/health/status signals are valid
  uint8_t last_counter;              // last counter value
  uint32_t last_timestamp;           // micro-s
  bool lagging;                      // true if and only if the time between updates is excessive
} RxStatus;

// params and flags about checksum, counter and frequency checks for each monitored address
typedef struct {
  const CanMsgCheck msg[MAX_ADDR_CHECK_MSGS];  // check either messages (e.g. honda steer)
  RxStatus status;
} RxCheck;

typedef struct {
  RxCheck *rx_checks;
  int rx_checks_len;
  const CanMsg *tx_msgs;
  int tx_msgs_len;
} safety_config;

typedef uint32_t (*get_checksum_t)(const CANPacket_t *to_push);
typedef uint32_t (*compute_checksum_t)(const CANPacket_t *to_push);
typedef uint8_t (*get_counter_t)(const CANPacket_t *to_push);
typedef bool (*get_quality_flag_valid_t)(const CANPacket_t *to_push);

typedef safety_config (*safety_hook_init)(uint16_t param);
typedef void (*rx_hook)(const CANPacket_t *to_push);
typedef bool (*tx_hook)(const CANPacket_t *to_send);
typedef int (*fwd_hook)(int bus_num, int addr);

typedef struct {
  safety_hook_init init;
  rx_hook rx;
  tx_hook tx;
  fwd_hook fwd;
  get_checksum_t get_checksum;
  compute_checksum_t compute_checksum;
  get_counter_t get_counter;
  get_quality_flag_valid_t get_quality_flag_valid;
} safety_hooks;

bool safety_rx_hook(const CANPacket_t *to_push);
bool safety_tx_hook(CANPacket_t *to_send);
uint32_t get_ts_elapsed(uint32_t ts, uint32_t ts_last);
int to_signed(int d, int bits);
void update_sample(struct sample_t *sample, int sample_new);
bool get_longitudinal_allowed(void);
int ROUND(float val);
void gen_crc_lookup_table_8(uint8_t poly, uint8_t crc_lut[]);
#ifdef CANFD
void gen_crc_lookup_table_16(uint16_t poly, uint16_t crc_lut[]);
#endif
void generic_rx_checks(bool stock_ecu_detected);
bool steer_torque_cmd_checks(int desired_torque, int steer_req, const SteeringLimits limits);
bool steer_angle_cmd_checks(int desired_angle, bool steer_control_enabled, const SteeringLimits limits);
bool longitudinal_accel_checks(int desired_accel, const LongitudinalLimits limits);
bool longitudinal_speed_checks(int desired_speed, const LongitudinalLimits limits);
bool longitudinal_gas_checks(int desired_gas, const LongitudinalLimits limits);
bool longitudinal_transmission_rpm_checks(int desired_transmission_rpm, const LongitudinalLimits limits);
bool longitudinal_brake_checks(int desired_brake, const LongitudinalLimits limits);
void pcm_cruise_check(bool cruise_engaged);

void safety_tick(const safety_config *safety_config);

// This can be set by the safety hooks
extern bool controls_allowed;
extern bool relay_malfunction;
extern bool gas_pressed;
extern bool gas_pressed_prev;
extern bool brake_pressed;
extern bool brake_pressed_prev;
extern bool regen_braking;
extern bool regen_braking_prev;
extern bool cruise_engaged_prev;
extern struct sample_t vehicle_speed;
extern bool vehicle_moving;
extern bool acc_main_on; // referred to as "ACC off" in ISO 15622:2018
extern int cruise_button_prev;
extern bool safety_rx_checks_invalid;

// for safety modes with torque steering control
extern int desired_torque_last;       // last desired steer torque
extern int rt_torque_last;            // last desired torque for real time check
extern int valid_steer_req_count;     // counter for steer request bit matching non-zero torque
extern int invalid_steer_req_count;   // counter to allow multiple frames of mismatching torque request bit
extern struct sample_t torque_meas;       // last 6 motor torques produced by the eps
extern struct sample_t torque_driver;     // last 6 driver torques measured
extern uint32_t ts_torque_check_last;
extern uint32_t ts_steer_req_mismatch_last;  // last timestamp steer req was mismatched with torque

// state for controls_allowed timeout logic
extern bool heartbeat_engaged;             // openpilot enabled, passed in heartbeat USB command
extern uint32_t heartbeat_engaged_mismatches;  // count of mismatches between heartbeat_engaged and controls_allowed

// for safety modes with angle steering control
extern uint32_t ts_angle_last;
extern int desired_angle_last;
extern struct sample_t angle_meas;         // last 6 steer angles/curvatures

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

// This flag allows AEB to be commanded from openpilot.
#define ALT_EXP_ALLOW_AEB 16

extern int alternative_experience;

// time since safety mode has been changed
extern uint32_t safety_mode_cnt;

typedef struct {
  uint16_t id;
  const safety_hooks *hooks;
} safety_hook_config;

extern uint16_t current_safety_mode;
extern uint16_t current_safety_param;
extern safety_config current_safety_config;

int safety_fwd_hook(int bus_num, int addr);
int set_safety_hooks(uint16_t mode, uint16_t param);

extern const safety_hooks body_hooks;
extern const safety_hooks chrysler_hooks;
extern const safety_hooks elm327_hooks;
extern const safety_hooks nooutput_hooks;
extern const safety_hooks alloutput_hooks;
extern const safety_hooks ford_hooks;
extern const safety_hooks gm_hooks;
extern const safety_hooks honda_nidec_hooks;
extern const safety_hooks honda_bosch_hooks;
extern const safety_hooks hyundai_canfd_hooks;
extern const safety_hooks hyundai_hooks;
extern const safety_hooks hyundai_legacy_hooks;
extern const safety_hooks mazda_hooks;
extern const safety_hooks nissan_hooks;
extern const safety_hooks subaru_hooks;
extern const safety_hooks subaru_preglobal_hooks;
extern const safety_hooks tesla_hooks;
extern const safety_hooks toyota_hooks;
extern const safety_hooks volkswagen_mqb_hooks;
extern const safety_hooks volkswagen_pq_hooks;
