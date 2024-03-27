extern uint8_t hw_type;
void can_send_msg(uint32_t addr, uint32_t dhr, uint32_t dlr, uint8_t len);

uint8_t uid[10];
uint32_t uds_engine_request = 0;
uint32_t uds_debug_request = 0;
uint8_t knee_detected = 0;
uint8_t sep_time = 0;

void process_uds(uint32_t addr, uint32_t dlr) {
  memcpy(uid, (void *)0x1FFF7A10U, 0xAU);

  if ((hw_type == HW_TYPE_BASE) &&
     ((addr == BROADCAST_ADDR) ||
      (addr == FALLBACK_ADDR))) { // OBD2 broadcast request, redirect to UDS?
    switch(dlr) {
      // VIN 09 OBD2
      case 0x020902U:
        can_send_msg(FALLBACK_R_ADDR, 0x4D4F4301U, 0x02491410U, 8U);
        uds_engine_request = 0xF190U;
        break;
      // VIN : F190 on broadcast
      case 0x90F12203U:
        can_send_msg(FALLBACK_R_ADDR, 0x4D4F4390U, 0xF1621410U, 8U);
        break;
      // VIN continue
      default:
        if ((dlr & 0xFF) == 0x30U) {
          sep_time = (dlr >> 16U) & 0xFF;
          delay(sep_time);
          can_send_msg(FALLBACK_R_ADDR, 0x5659444FU, 0x42414D21U, 8U);
          can_send_msg(FALLBACK_R_ADDR, 0x314E4F49U, 0x53524522U, 8U);
        }
        break;
    }
  } else if (addr == (ENGINE_ADDR + board.uds_offset)) { // UDS request to "main" ECU
    switch(dlr) {
      // TESTER PRESENT
      case 0x3E02U:
        can_send_msg(ENGINE_R_ADDR + board.uds_offset, 0x0U, 0x7E02U, 8U);
        break;
      // DIAGNOSTIC SESSION CONTROL: DEFAULT
      case 0x011002U:
        can_send_msg(ENGINE_R_ADDR + board.uds_offset, 0x0U, 0x015002U, 8U);
        break;
      // DIAGNOSTIC SESSION CONTROL: EXTENDED
      case 0x031002U:
        can_send_msg(ENGINE_R_ADDR + board.uds_offset, 0x0U, 0x035002U, 8U);
        break;
      // APPLICATION SOFTWARE IDENTIFICATION : F181 (used for fingerprinting, firmware version)
      case 0x81F12203U:
        COMPILE_TIME_ASSERT(sizeof(version) == 6U);
        can_send_msg(ENGINE_R_ADDR + board.uds_offset, ((version[2] << 24U) | (version[1] << 16U) | (version[0] << 8U) | 0x81U), 0xF1620A10U, 8U);
        uds_engine_request = 0xF181U;
        break;
      // ECU SERIAL NUMBER : F18C
      case 0x8CF12203U:
        can_send_msg(ENGINE_R_ADDR + board.uds_offset, ((uid[2] << 24U) | (uid[1] << 16U) | (uid[0] << 8U) | 0x8CU), 0xF1620D10U, 8U);
        uds_engine_request = 0xF18CU;
        break;
      // VIN : F190
      case 0x90F12203U:
        can_send_msg(ENGINE_R_ADDR + board.uds_offset, 0x4D4F4390U, 0xF1621410U, 8U);
        uds_engine_request = 0xF190U;
        break;
      // FLOW CONTROL MESSAGE
      default:
        if ((dlr & 0xFF) == 0x30U) {
          sep_time = (dlr >> 16U) & 0xFF;
          delay(sep_time);
          switch(uds_engine_request) {
            // APPLICATION SOFTWARE IDENTIFICATION : F181
            case 0xF181U:
              can_send_msg(ENGINE_R_ADDR + board.uds_offset, (knee_detected + 0x61), ((version[5] << 24U) | (version[4] << 16U) | (version[3] << 8U) | 0x21U), 8U);
              uds_engine_request = 0;
              break;
            // ECU SERIAL NUMBER : F18C
            case 0xF18CU:
              can_send_msg(ENGINE_R_ADDR + board.uds_offset, ((uid[9] << 24U) | (uid[8] << 16U) | (uid[7]<< 8U) | uid[6]), ((uid[5] << 24U) | (uid[4] << 16U) | (uid[3] << 8U) | 0x21U), 8U);
              uds_engine_request = 0;
              break;
            // VIN : F190
            case 0xF190U:
              can_send_msg(ENGINE_R_ADDR + board.uds_offset, 0x5659444FU, 0x42414D21U, 8U);
              can_send_msg(ENGINE_R_ADDR + board.uds_offset, 0x314E4F49U, 0x53524522U, 8U);
              uds_engine_request = 0;
              break;
          }
        }
        break;
    }
  } else if (addr == (DEBUG_ADDR + board.uds_offset)) { // UDS request to "DEBUG" ECU
    switch(dlr) {
      // TESTER PRESENT
      case 0x3E02U:
        can_send_msg(DEBUG_R_ADDR + board.uds_offset, 0x0U, 0x7E02U, 8U);
        break;
      // DIAGNOSTIC SESSION CONTROL: DEFAULT
      case 0x011002U:
        can_send_msg(DEBUG_R_ADDR + board.uds_offset, 0x0U, 0x015002U, 8U);
        break;
      // DIAGNOSTIC SESSION CONTROL: EXTENDED
      case 0x031002U:
        can_send_msg(DEBUG_R_ADDR + board.uds_offset, 0x0U, 0x035002U, 8U);
        break;
      // APPLICATION SOFTWARE IDENTIFICATION : F181 (used for git hash logging)
      case 0x81F12203U:
        COMPILE_TIME_ASSERT(sizeof(gitversion) == 8U);
        can_send_msg((DEBUG_R_ADDR + board.uds_offset), ((gitversion[2] << 24U) | (gitversion[1] << 16U) | (gitversion[0] << 8U) | 0x81U), 0xF1620B10U, 8U);
        uds_debug_request = 0xF181U;
        break;
      default:
        if ((dlr & 0xFF) == 0x30U) {
          sep_time = (dlr >> 16U) & 0xFF;
          delay(sep_time);
          switch(uds_debug_request) {
            // APPLICATION SOFTWARE IDENTIFICATION : F181
            case 0xF181U:
              can_send_msg((DEBUG_R_ADDR + board.uds_offset), ((gitversion[7]<< 8U) | gitversion[6]), ((gitversion[5] << 24U) | (gitversion[4] << 16U) | (gitversion[3] << 8U) | 0x21U), 8U);
              uds_debug_request = 0;
              break;
          }
        }
        break;
    }
  }
}
