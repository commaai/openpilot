// ******************** Prototypes ********************
typedef void (*board_init)(void);
typedef void (*board_enable_can_transciever)(uint8_t transciever, bool enabled);
typedef void (*board_enable_can_transcievers)(bool enabled);
typedef void (*board_set_led)(uint8_t color, bool enabled);
typedef void (*board_set_usb_power_mode)(uint8_t mode);
typedef void (*board_set_esp_gps_mode)(uint8_t mode);
typedef void (*board_set_can_mode)(uint8_t mode);
typedef void (*board_usb_power_mode_tick)(uint64_t tcnt);
typedef bool (*board_check_ignition)(void);

struct board {
  const char *board_type;
  const harness_configuration *harness_config;
  board_init init;
  board_enable_can_transciever enable_can_transciever;
  board_enable_can_transcievers enable_can_transcievers;
  board_set_led set_led;
  board_set_usb_power_mode set_usb_power_mode;
  board_set_esp_gps_mode set_esp_gps_mode;
  board_set_can_mode set_can_mode;
  board_usb_power_mode_tick usb_power_mode_tick;
  board_check_ignition check_ignition;
};

// ******************* Definitions ********************
// These should match the enums in cereal/log.capnp and __init__.py
#define HW_TYPE_UNKNOWN 0U
#define HW_TYPE_WHITE_PANDA 1U
#define HW_TYPE_GREY_PANDA 2U
#define HW_TYPE_BLACK_PANDA 3U
#define HW_TYPE_PEDAL 4U

// LED colors
#define LED_RED 0U
#define LED_GREEN 1U
#define LED_BLUE 2U

// USB power modes (from cereal.log.health)
#define USB_POWER_NONE 0U
#define USB_POWER_CLIENT 1U
#define USB_POWER_CDP 2U
#define USB_POWER_DCP 3U

// ESP modes
#define ESP_GPS_DISABLED 0U
#define ESP_GPS_ENABLED 1U
#define ESP_GPS_BOOTMODE 2U

// CAN modes
#define CAN_MODE_NORMAL 0U
#define CAN_MODE_GMLAN_CAN2 1U
#define CAN_MODE_GMLAN_CAN3 2U
#define CAN_MODE_OBD_CAN2 3U

// ********************* Globals **********************
uint8_t usb_power_mode = USB_POWER_NONE;
