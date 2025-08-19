// ******************** Prototypes ********************
typedef void (*board_init)(void);
typedef void (*board_board_tick)(void);
typedef bool (*board_get_button)(void);
typedef void (*board_init_bootloader)(void);
typedef void (*board_set_panda_power)(bool enabled);
typedef void (*board_set_panda_individual_power)(uint8_t port_num, bool enabled);
typedef void (*board_set_ignition)(bool enabled);
typedef void (*board_set_individual_ignition)(uint8_t bitmask);
typedef void (*board_set_harness_orientation)(uint8_t orientation);
typedef void (*board_set_can_mode)(uint8_t mode);
typedef void (*board_enable_can_transceiver)(uint8_t transceiver, bool enabled);
typedef void (*board_enable_header_pin)(uint8_t pin_num, bool enabled);
typedef float (*board_get_channel_power)(uint8_t channel);
typedef uint16_t (*board_get_sbu_mV)(uint8_t channel, uint8_t sbu);

struct board {
  GPIO_TypeDef * const led_GPIO[3];
  const uint8_t led_pin[3];
  const uint8_t led_pwm_channels[3]; // leave at 0 to disable PWM
  const uint16_t avdd_mV;
  board_init init;
  board_board_tick board_tick;
  board_get_button get_button;
  board_init_bootloader init_bootloader;
  board_set_panda_power set_panda_power;
  board_set_panda_individual_power set_panda_individual_power;
  board_set_ignition set_ignition;
  board_set_individual_ignition set_individual_ignition;
  board_set_harness_orientation set_harness_orientation;
  board_set_can_mode set_can_mode;
  board_enable_can_transceiver enable_can_transceiver;
  board_enable_header_pin enable_header_pin;
  board_get_channel_power get_channel_power;
  board_get_sbu_mV get_sbu_mV;

  // TODO: shouldn't need these
  bool has_spi;
};

// ******************* Definitions ********************
#define HW_TYPE_UNKNOWN 0U
#define HW_TYPE_V2 2U

// CAN modes
#define CAN_MODE_NORMAL 0U
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
