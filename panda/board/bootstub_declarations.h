// ******************** Prototypes ********************
void print(const char *a){ UNUSED(a); }
void puth(uint8_t i){ UNUSED(i); }
void puth2(uint8_t i){ UNUSED(i); }
void puth4(uint8_t i){ UNUSED(i); }
void hexdump(const void *a, int l){ UNUSED(a); UNUSED(l); }
typedef struct board board;
typedef struct harness_configuration harness_configuration;
// No CAN support on bootloader
void can_flip_buses(uint8_t bus1, uint8_t bus2){UNUSED(bus1); UNUSED(bus2);}
void pwm_init(TIM_TypeDef *TIM, uint8_t channel);
void pwm_set(TIM_TypeDef *TIM, uint8_t channel, uint8_t percentage);
// No UART support in bootloader
typedef struct uart_ring {} uart_ring;
uart_ring uart_ring_som_debug;
void uart_init(uart_ring *q, int baud) { UNUSED(q); UNUSED(baud); }

// ********************* Globals **********************
uint8_t hw_type = 0;
const board *current_board;
