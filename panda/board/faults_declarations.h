#pragma once

#define FAULT_STATUS_NONE 0U
#define FAULT_STATUS_TEMPORARY 1U
#define FAULT_STATUS_PERMANENT 2U

// Fault types, matches cereal.log.PandaState.FaultType
#define FAULT_RELAY_MALFUNCTION             (1UL << 0)
#define FAULT_UNUSED_INTERRUPT_HANDLED      (1UL << 1)
#define FAULT_INTERRUPT_RATE_CAN_1          (1UL << 2)
#define FAULT_INTERRUPT_RATE_CAN_2          (1UL << 3)
#define FAULT_INTERRUPT_RATE_CAN_3          (1UL << 4)
#define FAULT_INTERRUPT_RATE_TACH           (1UL << 5)
#define FAULT_INTERRUPT_RATE_GMLAN          (1UL << 6)   // deprecated
#define FAULT_INTERRUPT_RATE_INTERRUPTS     (1UL << 7)
#define FAULT_INTERRUPT_RATE_SPI_DMA        (1UL << 8)
#define FAULT_INTERRUPT_RATE_SPI_CS         (1UL << 9)
#define FAULT_INTERRUPT_RATE_UART_1         (1UL << 10)
#define FAULT_INTERRUPT_RATE_UART_2         (1UL << 11)
#define FAULT_INTERRUPT_RATE_UART_3         (1UL << 12)
#define FAULT_INTERRUPT_RATE_UART_5         (1UL << 13)
#define FAULT_INTERRUPT_RATE_UART_DMA       (1UL << 14)
#define FAULT_INTERRUPT_RATE_USB            (1UL << 15)
#define FAULT_INTERRUPT_RATE_TIM1           (1UL << 16)
#define FAULT_INTERRUPT_RATE_TIM3           (1UL << 17)
#define FAULT_REGISTER_DIVERGENT            (1UL << 18)
#define FAULT_INTERRUPT_RATE_KLINE_INIT     (1UL << 19)
#define FAULT_INTERRUPT_RATE_CLOCK_SOURCE   (1UL << 20)
#define FAULT_INTERRUPT_RATE_TICK           (1UL << 21)
#define FAULT_INTERRUPT_RATE_EXTI           (1UL << 22)
#define FAULT_INTERRUPT_RATE_SPI            (1UL << 23)
#define FAULT_INTERRUPT_RATE_UART_7         (1UL << 24)
#define FAULT_SIREN_MALFUNCTION             (1UL << 25)
#define FAULT_HEARTBEAT_LOOP_WATCHDOG       (1UL << 26)
#define FAULT_INTERRUPT_RATE_SOUND_DMA      (1UL << 27)

// Permanent faults
#define PERMANENT_FAULTS 0U

extern uint8_t fault_status;
extern uint32_t faults;

void fault_occurred(uint32_t fault);
void fault_recovered(uint32_t fault);
