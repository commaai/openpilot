#include "opendbc/safety/board/faults_declarations.h"

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
