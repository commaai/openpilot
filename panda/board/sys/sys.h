#pragma once

// ******************** critical ********************

void enable_interrupts(void);
void disable_interrupts(void);

extern uint8_t global_critical_depth;

#ifndef ENTER_CRITICAL
#define ENTER_CRITICAL()                                      \
  __disable_irq();                                            \
  global_critical_depth += 1U;
#endif

#ifndef EXIT_CRITICAL
#define EXIT_CRITICAL()                                       \
  global_critical_depth -= 1U;                                \
  if ((global_critical_depth == 0U) && interrupts_enabled) {  \
    __enable_irq();                                           \
  }
#endif

// ******************** faults ********************

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
#define FAULT_INTERRUPT_RATE_INTERRUPTS     (1UL << 7)
#define FAULT_INTERRUPT_RATE_SPI_DMA        (1UL << 8)
#define FAULT_SELF_TEST_CPU                 (1UL << 9)
#define FAULT_SELF_TEST_RAM                 (1UL << 10)
#define FAULT_SELF_TEST_FLASH               (1UL << 11)
#define FAULT_ECC                           (1UL << 12)
#define FAULT_OUTPUT_READBACK               (1UL << 13)
#define FAULT_DMA                           (1UL << 14)
#define FAULT_INTERRUPT_RATE_USB            (1UL << 15)
#define FAULT_CLOCK                         (1UL << 16)
#define FAULT_INTERRUPT_MISSING             (1UL << 17)
#define FAULT_REGISTER_DIVERGENT            (1UL << 18)
#define FAULT_INTERRUPT_RATE_CLOCK_SOURCE   (1UL << 20)
#define FAULT_INTERRUPT_RATE_TICK           (1UL << 21)
#define FAULT_INTERRUPT_RATE_EXTI           (1UL << 22)
#define FAULT_INTERRUPT_RATE_SPI            (1UL << 23)
#define FAULT_ADC                           (1UL << 24)
#define FAULT_SIREN_MALFUNCTION             (1UL << 25)
#define FAULT_HEARTBEAT_LOOP_WATCHDOG       (1UL << 26)
#define FAULT_INTERRUPT_RATE_SOUND_DMA      (1UL << 27)
#define FAULT_MPU                           (1UL << 28)
#define FAULT_TIMER                         (1UL << 29)

// permanent faults put panda in the safe state until reset
// TODO: enable faults one by one once their fleet rate is checked
#define PERMANENT_FAULTS 0U

extern uint8_t fault_status;
extern uint32_t faults;

void fault_occurred(uint32_t fault);
void fault_occurred_at(uint32_t fault, uint32_t value, const char *site);
// a bit can be raised from several places, so each logs where it was raised
#define FAULT_STR(x) #x
#define FAULT_XSTR(x) FAULT_STR(x)
#define fault_occurred_value(fault, value) fault_occurred_at((fault), (value), __FILE__ ":" FAULT_XSTR(__LINE__))
void fault_recovered(uint32_t fault);
void fault_safe_state(void);

// ******************** power_saving ********************

extern bool power_save_enabled;

void set_power_save_state(bool enable);
