// ******************** Prototypes ********************
typedef void (*board_init)(void);
typedef void (*board_enable_can_transceiver)(uint8_t transceiver, bool enabled);
typedef void (*board_enable_can_transceivers)(bool enabled);
typedef void (*board_set_led)(uint8_t color, bool enabled);
typedef void (*board_set_usb_power_mode)(uint8_t mode);
typedef void (*board_set_gps_mode)(uint8_t mode);
typedef void (*board_set_can_mode)(uint8_t mode);
typedef void (*board_usb_power_mode_tick)(uint32_t uptime);
typedef bool (*board_check_ignition)(void);
typedef uint32_t (*board_read_current)(void);
typedef void (*board_set_ir_power)(uint8_t percentage);
typedef void (*board_set_fan_power)(uint8_t percentage);
typedef void (*board_set_phone_power)(bool enabled);
typedef void (*board_set_clock_source_mode)(uint8_t mode);
typedef void (*board_set_siren)(bool enabled);

struct board {
  const char *board_type;
  const harness_configuration *harness_config;
  board_init init;
  board_enable_can_transceiver enable_can_transceiver;
  board_enable_can_transceivers enable_can_transceivers;
  board_set_led set_led;
  board_set_usb_power_mode set_usb_power_mode;
  board_set_gps_mode set_gps_mode;
  board_set_can_mode set_can_mode;
  board_usb_power_mode_tick usb_power_mode_tick;
  board_check_ignition check_ignition;
  board_read_current read_current;
  board_set_ir_power set_ir_power;
  board_set_fan_power set_fan_power;
  board_set_phone_power set_phone_power;
  board_set_clock_source_mode set_clock_source_mode;
  board_set_siren set_siren;
};

// ******************* Definitions ********************
// These should match the enums in cereal/log.capnp and __init__.py
#define HW_TYPE_UNKNOWN 0U
#define HW_TYPE_WHITE_PANDA 1U
#define HW_TYPE_GREY_PANDA 2U
#define HW_TYPE_BLACK_PANDA 3U
#define HW_TYPE_PEDAL 4U
#define HW_TYPE_UNO 5U
#define HW_TYPE_DOS 6U

// LED colors
#define LED_RED 0U
#define LED_GREEN 1U
#define LED_BLUE 2U

// USB power modes (from cereal.log.health)
#define USB_POWER_NONE 0U
#define USB_POWER_CLIENT 1U
#define USB_POWER_CDP 2U
#define USB_POWER_DCP 3U

// GPS modes
#define GPS_DISABLED 0U
#define GPS_ENABLED 1U
#define GPS_BOOTMODE 2U

// CAN modes
#define CAN_MODE_NORMAL 0U
#define CAN_MODE_GMLAN_CAN2 1U
#define CAN_MODE_GMLAN_CAN3 2U
#define CAN_MODE_OBD_CAN2 3U

// ********************* Globals **********************
uint8_t usb_power_mode = USB_POWER_NONE;

// ************ Board function prototypes *************
bool board_has_gps(void);
bool board_has_gmlan(void);
bool board_has_obd(void);
bool board_has_lin(void);
bool board_has_rtc(void);
bool board_has_relay(void);
