// ******************** Prototypes ********************
void puts(const char *a);
void puth(unsigned int i);
void puth2(unsigned int i);
typedef struct board board;
typedef struct harness_configuration harness_configuration;
void can_flip_buses(uint8_t bus1, uint8_t bus2);
void can_set_obd(uint8_t harness_orientation, bool obd);

// ********************* Globals **********************
uint8_t hw_type = 0;
const board *current_board;
bool is_enumerated = 0;
uint32_t heartbeat_counter = 0;
uint32_t uptime_cnt = 0;
