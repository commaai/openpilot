#define FAULT_STATUS_NONE 0U
#define FAULT_STATUS_TEMPORARY 1U
#define FAULT_STATUS_PERMANENT 2U

// Fault types, matches cereal.log.PandaState.FaultType
#define FAULT_RELAY_MALFUNCTION             (1U << 0)
#define FAULT_UNUSED_INTERRUPT_HANDLED      (1U << 1)
#define FAULT_INTERRUPT_RATE_CAN_1          (1U << 2)
#define FAULT_INTERRUPT_RATE_CAN_2          (1U << 3)
#define FAULT_INTERRUPT_RATE_CAN_3          (1U << 4)
#define FAULT_INTERRUPT_RATE_TACH           (1U << 5)
#define FAULT_INTERRUPT_RATE_GMLAN          (1U << 6)
#define FAULT_INTERRUPT_RATE_INTERRUPTS     (1U << 7)
#define FAULT_INTERRUPT_RATE_SPI_DMA        (1U << 8)
#define FAULT_INTERRUPT_RATE_SPI_CS         (1U << 9)
#define FAULT_INTERRUPT_RATE_UART_1         (1U << 10)
#define FAULT_INTERRUPT_RATE_UART_2         (1U << 11)
#define FAULT_INTERRUPT_RATE_UART_3         (1U << 12)
#define FAULT_INTERRUPT_RATE_UART_5         (1U << 13)
#define FAULT_INTERRUPT_RATE_UART_DMA       (1U << 14)
#define FAULT_INTERRUPT_RATE_USB            (1U << 15)
#define FAULT_INTERRUPT_RATE_TIM1           (1U << 16)
#define FAULT_INTERRUPT_RATE_TIM3           (1U << 17)
#define FAULT_REGISTER_DIVERGENT            (1U << 18)
#define FAULT_INTERRUPT_RATE_KLINE_INIT     (1U << 19)
#define FAULT_INTERRUPT_RATE_CLOCK_SOURCE   (1U << 20)
#define FAULT_INTERRUPT_RATE_TICK           (1U << 21)
#define FAULT_INTERRUPT_RATE_EXTI           (1U << 22)
#define FAULT_INTERRUPT_RATE_SPI            (1U << 23)
#define FAULT_INTERRUPT_RATE_UART_7         (1U << 24)
#define FAULT_SIREN_MALFUNCTION             (1U << 25)
#define FAULT_HEARTBEAT_LOOP_WATCHDOG       (1U << 26)
#define FAULT_LOGGING_RATE_LIMIT            (1U << 27)

// Permanent faults
#define PERMANENT_FAULTS 0U

uint8_t fault_status = FAULT_STATUS_NONE;
uint32_t faults = 0U;

void fault_occurred(uint32_t fault) {
  if ((faults & fault) == 0U) {
    if ((PERMANENT_FAULTS & fault) != 0U) {
      print("Permanent fault occurred: 0x"); puth(fault); print("\n");
      fault_status = FAULT_STATUS_PERMANENT;
    } else {
      print("Temporary fault occurred: 0x"); puth(fault); print("\n");
      fault_status = FAULT_STATUS_TEMPORARY;
    }
  }
  faults |= fault;
}

void fault_recovered(uint32_t fault) {
  if ((PERMANENT_FAULTS & fault) == 0U) {
    faults &= ~fault;
  } else {
    print("Cannot recover from a permanent fault!\n");
  }
}
