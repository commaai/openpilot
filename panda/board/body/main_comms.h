void comms_endpoint2_write(const uint8_t *data, uint32_t len) {
  UNUSED(data);
  UNUSED(len);
}

int comms_control_handler(ControlPacket_t *req, uint8_t *resp) {
  unsigned int resp_len = 0;

  switch (req->request) {
    // **** 0xc1: get hardware type
    case 0xc1:
      resp[0] = hw_type;
      resp_len = 1;
      break;

    // **** 0xd1: enter bootloader mode
    case 0xd1:
      switch (req->param1) {
        case 0:
          print("-> entering bootloader\n");
          enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
          NVIC_SystemReset();
          break;
        case 1:
          print("-> entering softloader\n");
          enter_bootloader_mode = ENTER_SOFTLOADER_MAGIC;
          NVIC_SystemReset();
          break;
        default:
          print("Bootloader mode invalid\n");
          break;
      }
      break;

    // **** 0xd3/0xd4: signature bytes
    case 0xd3:
    case 0xd4: {
      uint8_t offset = (req->request == 0xd3) ? 0U : 64U;
      resp_len = 64U;
      char *code = (char *)_app_start;
      int code_len = _app_start[0];
      (void)memcpy(resp, &code[code_len + offset], resp_len);
      break;
    }

    // **** 0xd6: get version
    case 0xd6:
      (void)memcpy(resp, gitversion, sizeof(gitversion));
      resp_len = sizeof(gitversion) - 1U;
      break;

    // **** 0xd8: reset ST
    case 0xd8:
      NVIC_SystemReset();
      break;

    // **** 0xe0: set motor speed
    case 0xe0:
      motor_set_speed((uint8_t)req->param1, (int8_t)req->param2);
      break;

    // **** 0xe1: stop motor
    case 0xe1:
      motor_set_speed((uint8_t)req->param1, 0);
      break;

    // **** 0xe2: get motor encoder state
    case 0xe2: {
      uint8_t motor = (uint8_t)req->param1;
      int32_t position = motor_encoder_get_position(motor);
      float rpm = motor_encoder_get_speed_rpm(motor);
      int32_t rpm_milli = (int32_t)(rpm * 1000.0f);
      (void)memcpy(resp, &position, sizeof(position));
      (void)memcpy(resp + sizeof(position), &rpm_milli, sizeof(rpm_milli));
      resp_len = (uint8_t)(sizeof(position) + sizeof(rpm_milli));
      break;
    }

    // **** 0xe3: reset encoder position
    case 0xe3:
      motor_encoder_reset((uint8_t)req->param1);
      break;

    // **** 0xe4: set motor target speed (rpm * 0.1)
    case 0xe4: {
      uint8_t motor = (uint8_t)req->param1;
      float target_rpm = ((int16_t)req->param2) * 0.1f;
      motor_speed_controller_set_target_rpm(motor, target_rpm);
      break;
    }

    // **** 0xdd: get healthpacket and CANPacket versions
    case 0xdd:
      resp[0] = HEALTH_PACKET_VERSION;
      resp[1] = CAN_PACKET_VERSION;
      resp[2] = CAN_HEALTH_PACKET_VERSION;
      resp_len = 3U;
      break;

    default:
      // Ignore unhandled requests
      break;
  }
  return resp_len;
}
