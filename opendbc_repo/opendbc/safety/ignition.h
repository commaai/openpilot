#ifndef SAFETY_IGNITION_H
#define SAFETY_IGNITION_H

#include <stdbool.h>
#include <stdint.h>

#include "opendbc/safety/can.h"
#include "opendbc/safety/helpers.h"

// read and reset by panda's firmware
extern bool ignition_can;
extern uint32_t ignition_can_cnt;
void ignition_can_reset(void);
void ignition_can_hook(const CANPacket_t *msg);

// the counters are at most 4 bits, so this marks that no previous counter was seen
#define IGNITION_NO_COUNTER 0xFFU

bool ignition_can = false;
uint32_t ignition_can_cnt = 0U;

// previous message counter per ignition message, IGNITION_NO_COUNTER if none was seen yet
static uint8_t prev_counter_rivian = IGNITION_NO_COUNTER;
static uint8_t prev_counter_tesla = IGNITION_NO_COUNTER;
static uint8_t prev_counter_vw_meb = IGNITION_NO_COUNTER;
static uint8_t prev_counter_psa[3] = {IGNITION_NO_COUNTER, IGNITION_NO_COUNTER, IGNITION_NO_COUNTER};

void ignition_can_reset(void) {
  ignition_can = false;
  ignition_can_cnt = 0U;
  prev_counter_rivian = IGNITION_NO_COUNTER;
  prev_counter_tesla = IGNITION_NO_COUNTER;
  prev_counter_vw_meb = IGNITION_NO_COUNTER;
  for (int i = 0; i < 3; i++) {
    prev_counter_psa[i] = IGNITION_NO_COUNTER;
  }
}

void ignition_can_hook(const CANPacket_t *msg) {
  // GM exception
  if (msg_matches(msg, 0x1F1U, 0U, 8U)) {
    // SystemPowerMode (2=Run, 3=Crank Request)
    ignition_can = GET_BIT(msg, 1U);
    ignition_can_cnt = 0U;
  }

  // Rivian R1S/T GEN1 exception
  if (msg_matches(msg, 0x152U, 0U, 8U)) {
    // 0x152 overlaps with Subaru pre-global which has this bit as the high beam
    uint8_t counter = msg->data[1] & 0xFU;  // max is only 14

    if ((prev_counter_rivian != IGNITION_NO_COUNTER) && (counter == ((prev_counter_rivian + 1U) % 15U))) {
      // VDM_OutputSignals->VDM_EpasPowerMode
      ignition_can = ((msg->data[7] >> 4U) & 0x3U) == 1U;  // VDM_EpasPowerMode_Drive_On=1
      ignition_can_cnt = 0U;
    }
    prev_counter_rivian = counter;
  }

  // Tesla Model 3/Y exception
  if (msg_matches(msg, 0x221U, 0U, 8U)) {
    // 0x221 overlaps with Rivian which has random data on byte 0
    uint8_t counter = msg->data[6] >> 4;

    if ((prev_counter_tesla != IGNITION_NO_COUNTER) && (counter == ((prev_counter_tesla + 1U) % 16U))) {
      // VCFRONT_LVPowerState->VCFRONT_vehiclePowerState
      uint8_t power_state = (msg->data[0] >> 5U) & 0x3U;
      ignition_can = power_state == 0x3U;  // VEHICLE_POWER_STATE_DRIVE=3
      ignition_can_cnt = 0U;
    }
    prev_counter_tesla = counter;
  }

  // Mazda exception
  if (msg_matches(msg, 0x9EU, 0U, 8U)) {
    ignition_can = (msg->data[0] >> 5) == 0x6U;
    ignition_can_cnt = 0U;
  }

  // Volkswagen MEB exception
  if (msg_matches(msg, 0x3C0U, 0U, 4U)) {
    uint8_t counter = msg->data[1] & 0xFU;

    if ((prev_counter_vw_meb != IGNITION_NO_COUNTER) && (counter == ((prev_counter_vw_meb + 1U) % 16U))) {
      // Klemmen_Status_01->ZAS_Kl_15
      ignition_can = GET_BIT(msg, 17U);
      ignition_can_cnt = 0U;
    }
    prev_counter_vw_meb = counter;
  }

  // PSA exception, the BSI is on bus 2 and only reaches bus 0 while the relay is open
  if (msg_matches(msg, 0x432U, 0U, 8U) || msg_matches(msg, 0x432U, 2U, 8U)) {
    uint8_t counter = msg->data[1] & 0xFU;

    if ((prev_counter_psa[msg->bus] != IGNITION_NO_COUNTER) && (counter == ((prev_counter_psa[msg->bus] + 1U) % 16U))) {
      // IS_DAT_BSI_432->ETAT_PRINCIP_SEV (0=stop, 1=ignition on, 2=cranking)
      ignition_can = (msg->data[6] & 0x3U) != 0U;
      ignition_can_cnt = 0U;
    }
    prev_counter_psa[msg->bus] = counter;
  }
}

#endif  // SAFETY_IGNITION_H
