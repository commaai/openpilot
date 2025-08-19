#pragma once

// SAE 2284-3 : minimum 16 tq, SJW 3, sample point at 81.3%
#define CAN_QUANTA 16U
#define CAN_SEQ1 12U
#define CAN_SEQ2 3U
#define CAN_SJW  3U

#define CAN_PCLK 48000U
// 333 = 33.3 kbps
// 5000 = 500 kbps
#define can_speed_to_prescaler(x) (CAN_PCLK / CAN_QUANTA * 10U / (x))

#define CAN_NAME_FROM_CANIF(CAN_DEV) (((CAN_DEV)==CAN1) ? "CAN1" : (((CAN_DEV) == CAN2) ? "CAN2" : "CAN3"))

void print(const char *a);

// kbps multiplied by 10
#define SPEEDS_ARRAY_SIZE 8
extern const uint32_t speeds[SPEEDS_ARRAY_SIZE];
#define DATA_SPEEDS_ARRAY_SIZE 1
extern const uint32_t data_speeds[DATA_SPEEDS_ARRAY_SIZE]; // No separate data speed, dummy

bool llcan_set_speed(CAN_TypeDef *CANx, uint32_t speed, bool loopback, bool silent);
void llcan_irq_disable(const CAN_TypeDef *CANx);
void llcan_irq_enable(const CAN_TypeDef *CANx);
bool llcan_init(CAN_TypeDef *CANx);
void llcan_clear_send(CAN_TypeDef *CANx);
