#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "board/can.h"
#include "board/health.h"
#include "board/body/motor_control.h"
#include "board/drivers/can_common_declarations.h"
#include "opendbc/safety/declarations.h"

#define BODY_CAN_ADDR_MOTOR_SPEED      0x201U
#define BODY_CAN_MOTOR_SPEED_PERIOD_US 10000U
#define BODY_BUS_NUMBER                 0U

static struct {
  bool pending;
  int32_t left_target_deci_rpm;
  int32_t right_target_deci_rpm;
} body_can_command;

void body_can_send_motor_speeds(uint8_t bus, float left_speed_rpm, float right_speed_rpm) {
  CANPacket_t pkt;
  pkt.bus = bus;
  pkt.addr = BODY_CAN_ADDR_MOTOR_SPEED;
  pkt.returned = 0;
  pkt.rejected = 0;
  pkt.extended = 0;
  pkt.fd = 0;
  pkt.data_len_code = 8;
  int16_t left_speed_deci = left_speed_rpm * 10;
  int16_t right_speed_deci = right_speed_rpm * 10;
  pkt.data[0] = (uint8_t)((left_speed_deci >> 8) & 0xFFU);
  pkt.data[1] = (uint8_t)(left_speed_deci & 0xFFU);
  pkt.data[2] = (uint8_t)((right_speed_deci >> 8) & 0xFFU);
  pkt.data[3] = (uint8_t)(right_speed_deci & 0xFFU);
  pkt.data[4] = 0U;
  pkt.data[5] = 0U;
  pkt.data[6] = 0U;
  pkt.data[7] = 0U;
  can_set_checksum(&pkt);
  can_send(&pkt, bus, true);
}

void body_can_process_target(int16_t left_target_deci_rpm, int16_t right_target_deci_rpm) {
  body_can_command.left_target_deci_rpm = (int32_t)left_target_deci_rpm;
  body_can_command.right_target_deci_rpm = (int32_t)right_target_deci_rpm;
  body_can_command.pending = true;
}

void body_can_init(void) {
  body_can_command.pending = false;
  can_silent = false;
  can_loopback = false;
  (void)set_safety_hooks(SAFETY_BODY, 0U);
  set_gpio_output(GPIOD, 2U, 0); // Enable CAN transceiver
  can_init_all();
}

void body_can_periodic(uint32_t now) {
  if (body_can_command.pending) {
    body_can_command.pending = false;
    float left_target_rpm = ((float)body_can_command.left_target_deci_rpm) * 0.1f;
    float right_target_rpm = ((float)body_can_command.right_target_deci_rpm) * 0.1f;
    motor_speed_controller_set_target_rpm(BODY_MOTOR_LEFT, left_target_rpm);
    motor_speed_controller_set_target_rpm(BODY_MOTOR_RIGHT, right_target_rpm);
  }

  static uint32_t last_motor_speed_tx_us = 0;
  if ((now - last_motor_speed_tx_us) >= BODY_CAN_MOTOR_SPEED_PERIOD_US) {
    float left_speed_rpm = motor_encoder_get_speed_rpm(BODY_MOTOR_LEFT);
    float right_speed_rpm = motor_encoder_get_speed_rpm(BODY_MOTOR_RIGHT);
    body_can_send_motor_speeds(BODY_BUS_NUMBER, left_speed_rpm, right_speed_rpm);
    last_motor_speed_tx_us = now;
  }
}
