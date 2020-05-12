// ******************** Prototypes ********************
void puts(const char *a);
void puth(unsigned int i);
void puth2(unsigned int i);
typedef struct board board;
typedef struct harness_configuration harness_configuration;

// ********************* Globals **********************
uint8_t hw_type = 0;
const board *current_board;
bool is_enumerated = 0;