#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "board/can.h"
#include "board/health.h"
#include "board/body/boards/board_declarations.h"
#include "board/drivers/drivers.h"
#include "opendbc/safety/declarations.h"
#include "board/body/bldc/bldc.h"

#define BODY_CAN_ADDR_MOTOR_SPEED        0x201U
#define BODY_CAN_ADDR_VAR_VALUES         0x202U
#define BODY_CAN_ADDR_BODY_DATA          0x203U
#define BODY_CAN_ADDR_V2_ID              0x222U
#define BODY_CAN_MOTOR_SPEED_PERIOD_US   10000U
#define BODY_CAN_CMD_TIMEOUT_US          100000U
#define BODY_BUS_NUMBER                  0U

static uint32_t last_can_cmd_timestamp_us = 0U;
static uint16_t counter = 0U;

void body_can_send_motor_speeds(uint8_t bus, float left_speed_rpm, float right_speed_rpm) {
  CANPacket_t pkt = {0};
  pkt.bus = bus;
  pkt.addr = BODY_CAN_ADDR_MOTOR_SPEED;
  pkt.data_len_code = 8;
  int16_t left_speed_deci = left_speed_rpm;
  int16_t right_speed_deci = -(right_speed_rpm);
  pkt.data[0] = (uint8_t)((left_speed_deci >> 8) & 0xFFU);
  pkt.data[1] = (uint8_t)(left_speed_deci & 0xFFU);
  pkt.data[2] = (uint8_t)((right_speed_deci >> 8) & 0xFFU);
  pkt.data[3] = (uint8_t)(right_speed_deci & 0xFFU);
  pkt.data[4] = 0U;
  pkt.data[5] = 0U;
  pkt.data[6] = counter & 0xFFU;
  can_set_checksum(&pkt);
  can_send(&pkt, bus, true);
  counter++;
}

void body_can_send_var_values(uint8_t bus, bool ignition, bool enable_motors, uint8_t fault, uint8_t left_z_errcode, uint8_t right_z_errcode) {
  CANPacket_t pkt = {0};
  pkt.bus = bus;
  pkt.addr = BODY_CAN_ADDR_VAR_VALUES;
  pkt.data_len_code = 3;
  pkt.data[0] = (ignition ? 1U : 0U) | ((enable_motors ? 1U : 0U) << 1U) | ((fault & 0x3FU) << 2U);
  pkt.data[1] = left_z_errcode;
  pkt.data[2] = right_z_errcode;
  can_set_checksum(&pkt);
  can_send(&pkt, bus, true);
}

void body_can_send_body_data(uint8_t bus, uint8_t mcu_temp_raw, uint16_t batt_voltage_raw, uint8_t batt_percentage, bool charger_connected) {
  CANPacket_t pkt = {0};
  pkt.bus = bus;
  pkt.addr = BODY_CAN_ADDR_BODY_DATA;
  pkt.data_len_code = 4;
  pkt.data[0] = mcu_temp_raw;
  pkt.data[1] = (uint8_t)((batt_voltage_raw >> 8) & 0xFFU);
  pkt.data[2] = (uint8_t)(batt_voltage_raw & 0xFFU);
  pkt.data[3] = (charger_connected ? 1U : 0U) | ((batt_percentage & 0x7FU) << 1U);
  can_set_checksum(&pkt);
  can_send(&pkt, bus, true);
}

void body_can_process_target(int16_t left_target_deci_rpm, int16_t right_target_deci_rpm) {
  rpm_left = (int)(((float)left_target_deci_rpm) * 0.1f);
  rpm_right = (int)(((float)right_target_deci_rpm) * 0.1f);
  last_can_cmd_timestamp_us = microsecond_timer_get();
}

void body_can_rx(CANPacket_t *msg) {
  if ((msg->addr == 0x250U) && (GET_LEN(msg) >= 4U)) {
    int16_t left_target_deci_rpm = (int16_t)((msg->data[0] << 8U) | msg->data[1]);
    int16_t right_target_deci_rpm = (int16_t)((msg->data[2] << 8U) | msg->data[3]);
    body_can_process_target(left_target_deci_rpm, right_target_deci_rpm);
  }
}

void body_can_init(void) {
  last_can_cmd_timestamp_us = 0U;
  can_silent = false;
  can_loopback = false;
  (void)set_safety_hooks(SAFETY_BODY, 0U);
  set_gpio_output(CAN_TRANSCEIVER_EN_PORT, CAN_TRANSCEIVER_EN_PIN, 0); // Enable CAN transceiver
  can_init_all();
}

void body_can_periodic(uint32_t now, bool ignition, bool plug_charging) {
  if ((last_can_cmd_timestamp_us != 0U) &&
      ((now - last_can_cmd_timestamp_us) >= BODY_CAN_CMD_TIMEOUT_US)) {
    rpm_left = 0;
    rpm_right = 0;
    last_can_cmd_timestamp_us = 0U;
  }

  static uint32_t last_motor_speed_tx_us = 0;
  if ((now - last_motor_speed_tx_us) >= BODY_CAN_MOTOR_SPEED_PERIOD_US) {
    float left_speed_rpm = motor_encoder_get_speed_rpm(BODY_MOTOR_LEFT);
    float right_speed_rpm = motor_encoder_get_speed_rpm(BODY_MOTOR_RIGHT);
    body_can_send_motor_speeds(BODY_BUS_NUMBER, left_speed_rpm, right_speed_rpm);
    body_can_send_var_values(BODY_BUS_NUMBER, ignition, enable_motors, 0U, rtY_Left.z_errCode, rtY_Right.z_errCode);
    body_can_send_body_data(BODY_BUS_NUMBER, 0U, batt_voltage_raw, batt_percentage, plug_charging);

    // Send message on 0x222 to identify as body v2
    CANPacket_t id_pkt = {0};
    id_pkt.bus = BODY_BUS_NUMBER;
    id_pkt.addr = BODY_CAN_ADDR_V2_ID;
    id_pkt.data_len_code = 1;
    id_pkt.data[0] = 1U;
    can_set_checksum(&id_pkt);
    can_send(&id_pkt, BODY_BUS_NUMBER, true);

    last_motor_speed_tx_us = now;
  }
}
