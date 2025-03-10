#pragma once

#define FAULT_STATUS_NONE 0U
#define FAULT_STATUS_TEMPORARY 1U
#define FAULT_STATUS_PERMANENT 2U

// Fault types, excerpt from cereal.log.PandaState.FaultType for safety tests
#define FAULT_RELAY_MALFUNCTION             (1UL << 0)
// ...

// Permanent faults
#define PERMANENT_FAULTS 0U

extern uint8_t fault_status;
extern uint32_t faults;

void fault_occurred(uint32_t fault);
void fault_recovered(uint32_t fault);
