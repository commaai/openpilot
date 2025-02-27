#pragma once

#define CLOCK_SOURCE_PERIOD_MS           50U
#define CLOCK_SOURCE_PULSE_LEN_MS        2U

void clock_source_set_period(uint8_t period);
void clock_source_init(void);
