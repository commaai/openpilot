#pragma once

#include "board/sys/sys.h"

uint8_t fault_status = FAULT_STATUS_NONE;
uint32_t faults = 0U;

void fault_occurred(uint32_t fault) {
  if ((faults & fault) == 0U) {
    if ((PERMANENT_FAULTS & fault) != 0U) {
      print("Permanent fault occurred: 0x"); puth(fault); print("\n");
      fault_status = FAULT_STATUS_PERMANENT;
      fault_safe_state();
    } else {
      print("Temporary fault occurred: 0x"); puth(fault); print("\n");
      fault_status = (fault_status == FAULT_STATUS_PERMANENT) ? FAULT_STATUS_PERMANENT : FAULT_STATUS_TEMPORARY;
    }
  }
  faults |= fault;
}

#define FAULT_SITES 32U

// printed once per call site
void fault_occurred_at(uint32_t fault, uint32_t value, const char *site) {
  static const char *sites[FAULT_SITES];
  static uint8_t site_cnt = 0U;
  bool seen = false;
  for (uint8_t i = 0U; i < site_cnt; i++) {
    seen = seen || (sites[i] == site);
  }
  if (!seen && (site_cnt < FAULT_SITES)) {
    sites[site_cnt] = site;
    site_cnt++;
    print("Fault 0x"); puth(fault); print(" at "); print(site); print(" value 0x"); puth(value); print("\n");
  }
  fault_occurred(fault);
}

void fault_recovered(uint32_t fault) {
  if ((PERMANENT_FAULTS & fault) == 0U) {
    faults &= ~fault;
  } else {
    print("Cannot recover from a permanent fault!\n");
  }
}
