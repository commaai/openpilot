// ******************** Prototypes ********************
typedef void (*board_init)(void);
typedef void (*board_set_led)(uint8_t color, bool enabled);
typedef void (*board_board_tick)(void);
typedef bool (*board_get_button)(void);
typedef void (*board_set_panda_power)(bool enabled);
typedef void (*board_set_ignition)(bool enabled);
typedef void (*board_set_individual_ignition)(uint8_t bitmask);
typedef void (*board_set_harness_orientation)(uint8_t orientation);
typedef void (*board_set_can_mode)(uint8_t mode);
typedef void (*board_enable_can_transciever)(uint8_t transciever, bool enabled);
typedef void (*board_enable_header_pin)(uint8_t pin_num, bool enabled);
typedef float (*board_get_channel_power)(uint8_t channel);
typedef uint16_t (*board_get_sbu_mV)(uint8_t channel, uint8_t sbu);

struct board {
  const char *board_type;
  const bool has_canfd;
  const bool has_sbu_sense;
  const uint16_t avdd_mV;
  board_init init;
  board_set_led set_led;
  board_board_tick board_tick;
  board_get_button get_button;
  board_set_panda_power set_panda_power;
  board_set_ignition set_ignition;
  board_set_individual_ignition set_individual_ignition;
  board_set_harness_orientation set_harness_orientation;
  board_set_can_mode set_can_mode;
  board_enable_can_transciever enable_can_transciever;
  board_enable_header_pin enable_header_pin;
  board_get_channel_power get_channel_power;
  board_get_sbu_mV get_sbu_mV;

  // TODO: shouldn't need these
  bool has_spi;
  bool has_hw_gmlan;
};

// ******************* Definitions ********************
#define HW_TYPE_UNKNOWN 0U
#define HW_TYPE_V1 1U
#define HW_TYPE_V2 2U

// LED colors
#define LED_RED 0U
#define LED_GREEN 1U
#define LED_BLUE 2U

// CAN modes
#define CAN_MODE_NORMAL 0U
#define CAN_MODE_GMLAN_CAN2 1U
#define CAN_MODE_GMLAN_CAN3 2U
#define CAN_MODE_OBD_CAN2 3U

// Harness states
#define HARNESS_ORIENTATION_NONE 0U
#define HARNESS_ORIENTATION_1 1U
#define HARNESS_ORIENTATION_2 2U

#define SBU1 0U
#define SBU2 1U

// ********************* Globals **********************
uint8_t harness_orientation = HARNESS_ORIENTATION_NONE;
uint8_t can_mode = CAN_MODE_NORMAL;
uint8_t ignition = 0U;


void unused_set_individual_ignition(uint8_t bitmask) {
  UNUSED(bitmask);
}

void unused_board_enable_header_pin(uint8_t pin_num, bool enabled) {
  UNUSED(pin_num);
  UNUSED(enabled);
}
