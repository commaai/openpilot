// ******************** Prototypes ********************
void puts(const char *a);
void puth(unsigned int i);
void puth2(unsigned int i);
void puth4(unsigned int i);
typedef struct board board;
typedef struct harness_configuration harness_configuration;
void can_flip_buses(uint8_t bus1, uint8_t bus2);
void pwm_init(TIM_TypeDef *TIM, uint8_t channel);
void pwm_set(TIM_TypeDef *TIM, uint8_t channel, uint8_t percentage);

// ********************* Globals **********************
uint8_t hw_type = 0;
const board *current_board;
bool is_enumerated = 0;
uint32_t uptime_cnt = 0;
bool green_led_enabled = false;

// heartbeat state
uint32_t heartbeat_counter = 0;
bool heartbeat_lost = false;
bool heartbeat_disabled = false;            // set over USB
bool heartbeat_engaged = false;             // openpilot enabled, passed in heartbeat USB command
uint32_t heartbeat_engaged_mismatches = 0;  // count of mismatches between heartbeat_engaged and controls_allowed

// siren state
bool siren_enabled = false;
uint32_t siren_countdown = 0; // siren plays while countdown > 0
uint32_t controls_allowed_countdown = 0;

