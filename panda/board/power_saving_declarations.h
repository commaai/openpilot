#pragma once

// WARNING: To stay in compliance with the SIL2 rules laid out in STM UM1840, we should never implement any of the available hardware low power modes.
// See rule: CoU_3

#define POWER_SAVE_STATUS_DISABLED 0
#define POWER_SAVE_STATUS_ENABLED 1

extern int power_save_status;

void set_power_save_state(int state);
