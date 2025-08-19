#pragma once

#include "opendbc/safety/safety_declarations.h"

// All common address checks except SCM_BUTTONS which isn't on one Nidec safety configuration
#define HONDA_COMMON_NO_SCM_FEEDBACK_RX_CHECKS(pt_bus)                                                                                      \
  {.msg = {{0x1A6, (pt_bus), 8, 25U, .max_counter = 3U, .ignore_quality_flag = true},                  /* SCM_BUTTONS */       \
           {0x296, (pt_bus), 4, 25U, .max_counter = 3U, .ignore_quality_flag = true}, { 0 }}},                                 \
  {.msg = {{0x158, (pt_bus), 8, 100U, .max_counter = 3U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  /* ENGINE_DATA */      \
  {.msg = {{0x17C, (pt_bus), 8, 100U, .max_counter = 3U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  /* POWERTRAIN_DATA */  \

#define HONDA_COMMON_RX_CHECKS(pt_bus)                                                                                                  \
  HONDA_COMMON_NO_SCM_FEEDBACK_RX_CHECKS(pt_bus)                                                                                        \
  {.msg = {{0x326, (pt_bus), 8, 10U, .max_counter = 3U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  /* SCM_FEEDBACK */  \

// Alternate brake message is used on some Honda Bosch, and Honda Bosch radarless (where PT bus is 0)
#define HONDA_ALT_BRAKE_ADDR_CHECK(pt_bus)                                                                                              \
  {.msg = {{0x1BE, (pt_bus), 3, 50U, .max_counter = 3U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  /* BRAKE_MODULE */  \

enum {
  HONDA_BTN_NONE = 0,
  HONDA_BTN_MAIN = 1,
  HONDA_BTN_CANCEL = 2,
  HONDA_BTN_SET = 3,
  HONDA_BTN_RESUME = 4,
};

static int honda_brake = 0;
static bool honda_brake_switch_prev = false;
static bool honda_alt_brake_msg = false;
static bool honda_fwd_brake = false;
static bool honda_bosch_long = false;
static bool honda_bosch_radarless = false;
static bool honda_bosch_canfd = false;
typedef enum {HONDA_NIDEC, HONDA_BOSCH} HondaHw;
static HondaHw honda_hw = HONDA_NIDEC;


static unsigned int honda_get_pt_bus(void) {
  return ((honda_hw == HONDA_BOSCH) && !honda_bosch_radarless && !honda_bosch_canfd) ? 1U : 0U;
}

static uint32_t honda_get_checksum(const CANPacket_t *msg) {
  int checksum_byte = GET_LEN(msg) - 1U;
  return (uint8_t)(msg->data[checksum_byte]) & 0xFU;
}

static uint32_t honda_compute_checksum(const CANPacket_t *msg) {
  int len = GET_LEN(msg);
  uint8_t checksum = 0U;
  unsigned int addr = msg->addr;
  while (addr > 0U) {
    checksum += (uint8_t)(addr & 0xFU); addr >>= 4;
  }
  for (int j = 0; j < len; j++) {
    uint8_t byte = msg->data[j];
    checksum += (uint8_t)(byte & 0xFU) + (byte >> 4U);
    if (j == (len - 1)) {
      checksum -= (byte & 0xFU);  // remove checksum in message
    }
  }
  return (uint8_t)((8U - checksum) & 0xFU);
}

static uint8_t honda_get_counter(const CANPacket_t *msg) {
  int counter_byte = GET_LEN(msg) - 1U;
  return (msg->data[counter_byte] >> 4U) & 0x3U;
}

static void honda_rx_hook(const CANPacket_t *msg) {
  const bool pcm_cruise = ((honda_hw == HONDA_BOSCH) && !honda_bosch_long) || (honda_hw == HONDA_NIDEC);
  unsigned int pt_bus = honda_get_pt_bus();

  // sample speed
  if (msg->addr == 0x158U) {
    vehicle_moving = msg->data[0] | msg->data[1];
  }

  // check ACC main state
  // 0x326 for all Bosch and some Nidec, 0x1A6 for some Nidec
  if ((msg->addr == 0x326U) || (msg->addr == 0x1A6U)) {
    acc_main_on = GET_BIT(msg, ((msg->addr == 0x326U) ? 28U : 47U));
    if (!acc_main_on) {
      controls_allowed = false;
    }
  }

  // enter controls when PCM enters cruise state
  if (pcm_cruise && (msg->addr == 0x17CU)) {
    const bool cruise_engaged = GET_BIT(msg, 38U);
    // engage on rising edge
    if (cruise_engaged && !cruise_engaged_prev) {
      controls_allowed = true;
    }

    // Since some Nidec cars can brake down to 0 after the PCM disengages,
    // we don't disengage when the PCM does.
    if (!cruise_engaged && (honda_hw != HONDA_NIDEC)) {
      controls_allowed = false;
    }
    cruise_engaged_prev = cruise_engaged;
  }

  // state machine to enter and exit controls for button enabling
  // 0x1A6 for the ILX, 0x296 for the Civic Touring
  if (((msg->addr == 0x1A6U) || (msg->addr == 0x296U)) && (msg->bus == pt_bus)) {
    int button = (msg->data[0] & 0xE0U) >> 5;

    // enter controls on the falling edge of set or resume
    bool set = (button != HONDA_BTN_SET) && (cruise_button_prev == HONDA_BTN_SET);
    bool res = (button != HONDA_BTN_RESUME) && (cruise_button_prev == HONDA_BTN_RESUME);
    if (acc_main_on && !pcm_cruise && (set || res)) {
      controls_allowed = true;
    }

    // exit controls once main or cancel are pressed
    if ((button == HONDA_BTN_MAIN) || (button == HONDA_BTN_CANCEL)) {
      controls_allowed = false;
    }
    cruise_button_prev = button;
  }

  // user brake signal on 0x17C reports applied brake from computer brake on accord
  // and crv, which prevents the usual brake safety from working correctly. these
  // cars have a signal on 0x1BE which only detects user's brake being applied so
  // in these cases, this is used instead.
  // most hondas: 0x17C
  // accord, crv: 0x1BE
  if (honda_alt_brake_msg) {
    if (msg->addr == 0x1BEU) {
      brake_pressed = GET_BIT(msg, 4U);
    }
  } else {
    if (msg->addr == 0x17CU) {
      // also if brake switch is 1 for two CAN frames, as brake pressed is delayed
      const bool brake_switch = GET_BIT(msg, 32U);
      brake_pressed = (GET_BIT(msg, 53U)) || (brake_switch && honda_brake_switch_prev);
      honda_brake_switch_prev = brake_switch;
    }
  }

  if (msg->addr == 0x17CU) {
    gas_pressed = msg->data[0] != 0U;
  }

  // disable stock Honda AEB in alternative experience
  if (!(alternative_experience & ALT_EXP_DISABLE_STOCK_AEB)) {
    if ((msg->bus == 2U) && (msg->addr == 0x1FAU)) {
      bool honda_stock_aeb = GET_BIT(msg, 29U);
      int honda_stock_brake = (msg->data[0] << 2) | (msg->data[1] >> 6);

      // Forward AEB when stock braking is higher than openpilot braking
      // only stop forwarding when AEB event is over
      if (!honda_stock_aeb) {
        honda_fwd_brake = false;
      } else if (honda_stock_brake >= honda_brake) {
        honda_fwd_brake = true;
      } else {
        // Leave Honda forward brake as is
      }
    }
  }
}

static bool honda_tx_hook(const CANPacket_t *msg) {

  const LongitudinalLimits HONDA_BOSCH_LONG_LIMITS = {
    .max_accel = 200,   // accel is used for brakes
    .min_accel = -350,

    .max_gas = 2000,
    .inactive_gas = -30000,
  };

  const LongitudinalLimits HONDA_NIDEC_LONG_LIMITS = {
    .max_gas = 198,  // 0xc6
    .max_brake = 255,

    .inactive_speed = 0,
  };

  bool tx = true;

  unsigned int bus_pt = honda_get_pt_bus();
  unsigned int bus_buttons = (honda_bosch_radarless) ? 2U : bus_pt;  // the camera controls ACC on radarless Bosch cars

  // ACC_HUD: safety check (nidec w/o pedal)
  if ((msg->addr == 0x30CU) && (msg->bus == bus_pt)) {
    int pcm_speed = (msg->data[0] << 8) | msg->data[1];
    int pcm_gas = msg->data[2];

    bool violation = false;
    violation |= longitudinal_speed_checks(pcm_speed, HONDA_NIDEC_LONG_LIMITS);
    violation |= longitudinal_gas_checks(pcm_gas, HONDA_NIDEC_LONG_LIMITS);
    if (violation) {
      tx = false;
    }
  }

  // BRAKE: safety check (nidec)
  if ((msg->addr == 0x1FAU) && (msg->bus == bus_pt)) {
    honda_brake = (msg->data[0] << 2) + ((msg->data[1] >> 6) & 0x3U);
    if (longitudinal_brake_checks(honda_brake, HONDA_NIDEC_LONG_LIMITS)) {
      tx = false;
    }
    if (honda_fwd_brake) {
      tx = false;
    }
  }

  // BRAKE/GAS: safety check (bosch)
  if ((msg->addr == 0x1DFU) && (msg->bus == bus_pt)) {
    int accel = (msg->data[3] << 3) | ((msg->data[4] >> 5) & 0x7U);
    accel = to_signed(accel, 11);

    int gas = (msg->data[0] << 8) | msg->data[1];
    gas = to_signed(gas, 16);

    bool violation = false;
    violation |= longitudinal_accel_checks(accel, HONDA_BOSCH_LONG_LIMITS);
    violation |= longitudinal_gas_checks(gas, HONDA_BOSCH_LONG_LIMITS);
    if (violation) {
      tx = false;
    }
  }

  // ACCEL: safety check (radarless)
  if ((msg->addr == 0x1C8U) && (msg->bus == bus_pt)) {
    int accel = (msg->data[0] << 4) | (msg->data[1] >> 4);
    accel = to_signed(accel, 12);

    bool violation = false;
    violation |= longitudinal_accel_checks(accel, HONDA_BOSCH_LONG_LIMITS);
    if (violation) {
      tx = false;
    }
  }

  // STEER: safety check
  if ((msg->addr == 0xE4U) || (msg->addr == 0x194U)) {
    if (!controls_allowed) {
      bool steer_applied = msg->data[0] | msg->data[1];
      if (steer_applied) {
        tx = false;
      }
    }
  }

  // Bosch supplemental control check
  if (msg->addr == 0xE5U) {
    if ((GET_BYTES(msg, 0, 4) != 0x10800004U) || ((GET_BYTES(msg, 4, 4) & 0x00FFFFFFU) != 0x0U)) {
      tx = false;
    }
  }

  // FORCE CANCEL: safety check only relevant when spamming the cancel button in Bosch HW
  // ensuring that only the cancel button press is sent (VAL 2) when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  if ((msg->addr == 0x296U) && !controls_allowed && (msg->bus == bus_buttons)) {
    if (((msg->data[0] >> 5) & 0x7U) != 2U) {
      tx = false;
    }
  }

  // Only tester present ("\x02\x3E\x80\x00\x00\x00\x00\x00") allowed on diagnostics address
  if (msg->addr == 0x18DAB0F1U) {
    if ((GET_BYTES(msg, 0, 4) != 0x00803E02U) || (GET_BYTES(msg, 4, 4) != 0x0U)) {
      tx = false;
    }
  }

  return tx;
}

static safety_config honda_nidec_init(uint16_t param) {
  // 0x1FA is dynamically forwarded based on stock AEB
  // 0xE4 is steering on all cars except CRV and RDX, 0x194 for CRV and RDX,
  // 0x1FA is brake control, 0x30C is acc hud, 0x33D is lkas hud
  static CanMsg HONDA_N_TX_MSGS[] = {{0xE4, 0, 5, .check_relay = true}, {0x194, 0, 4, .check_relay = true}, {0x1FA, 0, 8, .check_relay = false},
                                     {0x30C, 0, 8, .check_relay = true}, {0x33D, 0, 5, .check_relay = true}};

  const uint16_t HONDA_PARAM_NIDEC_ALT = 4;

  honda_hw = HONDA_NIDEC;
  honda_brake = 0;
  honda_brake_switch_prev = false;
  honda_fwd_brake = false;
  honda_alt_brake_msg = false;
  honda_bosch_long = false;
  honda_bosch_radarless = false;
  honda_bosch_canfd = false;

  safety_config ret;

  bool enable_nidec_alt = GET_FLAG(param, HONDA_PARAM_NIDEC_ALT);

  if (enable_nidec_alt) {
    // For Nidecs with main on signal on an alternate msg (missing 0x326)
    static RxCheck honda_nidec_alt_rx_checks[] = {
      HONDA_COMMON_NO_SCM_FEEDBACK_RX_CHECKS(0)
      {.msg = {{0x1FA, 2, 8, 50U, .max_counter = 3U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  // BRAKE_COMMAND
    };

    SET_RX_CHECKS(honda_nidec_alt_rx_checks, ret);
  } else {
    // Nidec includes BRAKE_COMMAND
    static RxCheck honda_nidec_common_rx_checks[] = {
      HONDA_COMMON_RX_CHECKS(0)
      {.msg = {{0x1FA, 2, 8, 50U, .max_counter = 3U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  // BRAKE_COMMAND
    };

    SET_RX_CHECKS(honda_nidec_common_rx_checks, ret);
  }

  SET_TX_MSGS(HONDA_N_TX_MSGS, ret);

  return ret;
}

static safety_config honda_bosch_init(uint16_t param) {
  static CanMsg HONDA_BOSCH_TX_MSGS[] = {{0xE4, 0, 5, .check_relay = true}, {0xE5, 0, 8, .check_relay = true}, {0x296, 1, 4, .check_relay = false},
                                         {0x33D, 0, 5, .check_relay = true}, {0x33D, 0, 8, .check_relay = true}, {0x33DA, 0, 5, .check_relay = true}, {0x33DB, 0, 8, .check_relay = true}};  // Bosch

  static CanMsg HONDA_BOSCH_LONG_TX_MSGS[] = {{0xE4, 1, 5, .check_relay = true}, {0x1DF, 1, 8, .check_relay = true}, {0x1EF, 1, 8, .check_relay = false},
                                              {0x1FA, 1, 8, .check_relay = false}, {0x30C, 1, 8, .check_relay = false}, {0x33D, 1, 5, .check_relay = true},
                                              {0x33DA, 1, 5, .check_relay = true}, {0x33DB, 1, 8, .check_relay = true}, {0x39F, 1, 8, .check_relay = false},
                                              {0x18DAB0F1, 1, 8, .check_relay = false}};  // Bosch w/ gas and brakes

  static CanMsg HONDA_RADARLESS_TX_MSGS[] = {{0xE4, 0, 5, .check_relay = true}, {0x296, 2, 4, .check_relay = false}, {0x33D, 0, 8, .check_relay = true}};  // Bosch radarless

  static CanMsg HONDA_RADARLESS_LONG_TX_MSGS[] = {{0xE4, 0, 5, .check_relay = true}, {0x33D, 0, 8, .check_relay = true}, {0x1C8, 0, 8, .check_relay = true},
                                                  {0x30C, 0, 8, .check_relay = true}};  // Bosch radarless w/ gas and brakes

  static CanMsg HONDA_CANFD_TX_MSGS[] = {{0xE4, 0, 5, .check_relay = true}, {0x296, 0, 4, .check_relay = false}, {0x33D, 0, 8, .check_relay = true}};


  const uint16_t HONDA_PARAM_ALT_BRAKE = 1;
  const uint16_t HONDA_PARAM_RADARLESS = 8;
  const uint16_t HONDA_PARAM_BOSCH_CANFD = 16;

  // Bosch radarless has the powertrain bus on bus 0
  static RxCheck honda_bosch_pt0_rx_checks[] = {
    HONDA_COMMON_RX_CHECKS(0)
  };

  static RxCheck honda_bosch_pt0_alt_brake_rx_checks[] = {
    HONDA_COMMON_RX_CHECKS(0)
    HONDA_ALT_BRAKE_ADDR_CHECK(0)
  };

  // Bosch has powertrain on bus 1, verified 0x1A6 does not exist
  static RxCheck honda_bosch_pt1_rx_checks[] = {
    HONDA_COMMON_RX_CHECKS(1)
  };

  static RxCheck honda_bosch_pt1_alt_brake_rx_checks[] = {
    HONDA_COMMON_RX_CHECKS(1)
    HONDA_ALT_BRAKE_ADDR_CHECK(1)
  };

  honda_hw = HONDA_BOSCH;
  honda_brake_switch_prev = false;
  honda_bosch_radarless = GET_FLAG(param, HONDA_PARAM_RADARLESS);
  honda_bosch_canfd = GET_FLAG(param, HONDA_PARAM_BOSCH_CANFD);
  // Checking for alternate brake override from safety parameter
  honda_alt_brake_msg = GET_FLAG(param, HONDA_PARAM_ALT_BRAKE);

  // radar disabled so allow gas/brakes
#ifdef ALLOW_DEBUG
  const uint16_t HONDA_PARAM_BOSCH_LONG = 2;
  honda_bosch_long = GET_FLAG(param, HONDA_PARAM_BOSCH_LONG);
#endif

  safety_config ret;
  if (honda_bosch_radarless || honda_bosch_canfd) {
    if (honda_alt_brake_msg) {
      SET_RX_CHECKS(honda_bosch_pt0_alt_brake_rx_checks, ret);
    } else {
      SET_RX_CHECKS(honda_bosch_pt0_rx_checks, ret);
    }
  } else {
   if (honda_alt_brake_msg) {
     SET_RX_CHECKS(honda_bosch_pt1_alt_brake_rx_checks, ret);
   } else {
     SET_RX_CHECKS(honda_bosch_pt1_rx_checks, ret);
   }
  }

  if (honda_bosch_radarless) {
    if (honda_bosch_long) {
      SET_TX_MSGS(HONDA_RADARLESS_LONG_TX_MSGS, ret);
    } else {
      SET_TX_MSGS(HONDA_RADARLESS_TX_MSGS, ret);
    }
  } else if (honda_bosch_canfd) {
    SET_TX_MSGS(HONDA_CANFD_TX_MSGS, ret);
  } else {
    if (honda_bosch_long) {
      SET_TX_MSGS(HONDA_BOSCH_LONG_TX_MSGS, ret);
    } else {
      SET_TX_MSGS(HONDA_BOSCH_TX_MSGS, ret);
    }
  }
  return ret;
}

static bool honda_nidec_fwd_hook(int bus_num, int addr) {
  bool block_msg = false;

  if (bus_num == 2) {
    // forwarded if stock AEB is active
    bool is_brake_msg = addr == 0x1FA;
    block_msg = is_brake_msg && !honda_fwd_brake;
  }

  return block_msg;
}

const safety_hooks honda_nidec_hooks = {
  .init = honda_nidec_init,
  .rx = honda_rx_hook,
  .tx = honda_tx_hook,
  .fwd = honda_nidec_fwd_hook,
  .get_counter = honda_get_counter,
  .get_checksum = honda_get_checksum,
  .compute_checksum = honda_compute_checksum,
};

const safety_hooks honda_bosch_hooks = {
  .init = honda_bosch_init,
  .rx = honda_rx_hook,
  .tx = honda_tx_hook,
  .get_counter = honda_get_counter,
  .get_checksum = honda_get_checksum,
  .compute_checksum = honda_compute_checksum,
};
