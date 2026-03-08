#pragma once

// SAE J2284-4 document specifies a bus-line network running at 2 Mbit/s
// SAE J2284-5 document specifies a point-to-point communication running at 5 Mbit/s

#define CAN_PCLK 80000U // KHz, sourced from PLL1Q
#define BITRATE_PRESCALER 2U // Valid from 250Kbps to 5Mbps with 80Mhz clock
#define CAN_SP_NOMINAL 80U // 80% for both SAE J2284-4 and SAE J2284-5
#define CAN_SP_DATA_2M 80U // 80% for SAE J2284-4
#define CAN_SP_DATA_5M 75U // 75% for SAE J2284-5
#define CAN_QUANTA(speed, prescaler) (CAN_PCLK / ((speed) / 10U * (prescaler)))
#define CAN_SEG1(tq, sp) (((tq) * (sp) / 100U)- 1U)
#define CAN_SEG2(tq, sp) ((tq) * (100U - (sp)) / 100U)

// FDCAN core settings
#define FDCAN_START_ADDRESS 0x4000AC00UL
#define FDCAN_OFFSET 3384UL // bytes for each FDCAN module, equally
#define FDCAN_OFFSET_W 846UL // words for each FDCAN module, equally

// FDCAN_RX_FIFO_0_EL_CNT + FDCAN_TX_FIFO_EL_CNT can't exceed 47 elements (47 * 72 bytes = 3,384 bytes) per FDCAN module

// RX FIFO 0
#define FDCAN_RX_FIFO_0_EL_CNT 46UL
#define FDCAN_RX_FIFO_0_HEAD_SIZE 8UL // bytes
#define FDCAN_RX_FIFO_0_DATA_SIZE 64UL // bytes
#define FDCAN_RX_FIFO_0_EL_SIZE (FDCAN_RX_FIFO_0_HEAD_SIZE + FDCAN_RX_FIFO_0_DATA_SIZE)
#define FDCAN_RX_FIFO_0_EL_W_SIZE (FDCAN_RX_FIFO_0_EL_SIZE / 4UL)
#define FDCAN_RX_FIFO_0_OFFSET 0UL

// TX FIFO
#define FDCAN_TX_FIFO_EL_CNT 1UL
#define FDCAN_TX_FIFO_HEAD_SIZE 8UL // bytes
#define FDCAN_TX_FIFO_DATA_SIZE 64UL // bytes
#define FDCAN_TX_FIFO_EL_SIZE (FDCAN_TX_FIFO_HEAD_SIZE + FDCAN_TX_FIFO_DATA_SIZE)
#define FDCAN_TX_FIFO_OFFSET (FDCAN_RX_FIFO_0_OFFSET + (FDCAN_RX_FIFO_0_EL_CNT * FDCAN_RX_FIFO_0_EL_W_SIZE))

#define CAN_NAME_FROM_CANIF(CAN_DEV) (((CAN_DEV)==FDCAN1) ? "FDCAN1" : (((CAN_DEV) == FDCAN2) ? "FDCAN2" : "FDCAN3"))
#define CAN_NUM_FROM_CANIF(CAN_DEV) (((CAN_DEV)==FDCAN1) ? 0UL : (((CAN_DEV) == FDCAN2) ? 1UL : 2UL))


void print(const char *a);

// kbps multiplied by 10
#define SPEEDS_ARRAY_SIZE 8
extern const uint32_t speeds[SPEEDS_ARRAY_SIZE];
#define DATA_SPEEDS_ARRAY_SIZE 10
extern const uint32_t data_speeds[DATA_SPEEDS_ARRAY_SIZE];

bool llcan_set_speed(FDCAN_GlobalTypeDef *FDCANx, uint32_t speed, uint32_t data_speed, bool non_iso, bool loopback, bool silent);
void llcan_irq_disable(const FDCAN_GlobalTypeDef *FDCANx);
void llcan_irq_enable(const FDCAN_GlobalTypeDef *FDCANx);
bool llcan_init(FDCAN_GlobalTypeDef *FDCANx);
void llcan_clear_send(FDCAN_GlobalTypeDef *FDCANx);
