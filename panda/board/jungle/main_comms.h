extern int _app_start[0xc000]; // Only first 3 sectors of size 0x4000 are used

int get_jungle_health_pkt(void *dat) {
  COMPILE_TIME_ASSERT(sizeof(struct jungle_health_t) <= USBPACKET_MAX_SIZE);
  struct jungle_health_t * health = (struct jungle_health_t*)dat;

  health->uptime_pkt = uptime_cnt;
  health->ch1_power = current_board->get_channel_power(1U);
  health->ch2_power = current_board->get_channel_power(2U);
  health->ch3_power = current_board->get_channel_power(3U);
  health->ch4_power = current_board->get_channel_power(4U);
  health->ch5_power = current_board->get_channel_power(5U);
  health->ch6_power = current_board->get_channel_power(6U);

  health->ch1_sbu1_mV = current_board->get_sbu_mV(1U, SBU1);
  health->ch1_sbu2_mV = current_board->get_sbu_mV(1U, SBU2);
  health->ch2_sbu1_mV = current_board->get_sbu_mV(2U, SBU1);
  health->ch2_sbu2_mV = current_board->get_sbu_mV(2U, SBU2);
  health->ch3_sbu1_mV = current_board->get_sbu_mV(3U, SBU1);
  health->ch3_sbu2_mV = current_board->get_sbu_mV(3U, SBU2);
  health->ch4_sbu1_mV = current_board->get_sbu_mV(4U, SBU1);
  health->ch4_sbu2_mV = current_board->get_sbu_mV(4U, SBU2);
  health->ch5_sbu1_mV = current_board->get_sbu_mV(5U, SBU1);
  health->ch5_sbu2_mV = current_board->get_sbu_mV(5U, SBU2);
  health->ch6_sbu1_mV = current_board->get_sbu_mV(6U, SBU1);
  health->ch6_sbu2_mV = current_board->get_sbu_mV(6U, SBU2);

  return sizeof(*health);
}

// send on serial, first byte to select the ring
void comms_endpoint2_write(const uint8_t *data, uint32_t len) {
  UNUSED(data);
  UNUSED(len);
}

int comms_control_handler(ControlPacket_t *req, uint8_t *resp) {
  unsigned int resp_len = 0;
  uint32_t time;

#ifdef DEBUG_COMMS
  print("raw control request: "); hexdump(req, sizeof(ControlPacket_t)); print("\n");
  print("- request "); puth(req->request); print("\n");
  print("- param1 "); puth(req->param1); print("\n");
  print("- param2 "); puth(req->param2); print("\n");
#endif

  switch (req->request) {
    // **** 0xa0: Set panda power.
    case 0xa0:
      current_board->set_panda_power((req->param1 == 1U));
      break;
    // **** 0xa1: Set harness orientation.
    case 0xa1:
      current_board->set_harness_orientation(req->param1);
      break;
    // **** 0xa2: Set ignition.
    case 0xa2:
      current_board->set_ignition((req->param1 == 1U));
      break;
    // **** 0xa0: Set panda power per channel by bitmask.
    case 0xa3:
      current_board->set_panda_individual_power(req->param1, (req->param2 > 0U));
      break;
    // **** 0xa8: get microsecond timer
    case 0xa8:
      time = microsecond_timer_get();
      resp[0] = (time & 0x000000FFU);
      resp[1] = ((time & 0x0000FF00U) >> 8U);
      resp[2] = ((time & 0x00FF0000U) >> 16U);
      resp[3] = ((time & 0xFF000000U) >> 24U);
      resp_len = 4U;
      break;
    // **** 0xc0: reset communications
    case 0xc0:
      comms_can_reset();
      break;
    // **** 0xc1: get hardware type
    case 0xc1:
      resp[0] = hw_type;
      resp_len = 1;
      break;
    // **** 0xc2: CAN health stats
    case 0xc2:
      COMPILE_TIME_ASSERT(sizeof(can_health_t) <= USBPACKET_MAX_SIZE);
      if (req->param1 < 3U) {
        update_can_health_pkt(req->param1, 0U);
        can_health[req->param1].can_speed = (bus_config[req->param1].can_speed / 10U);
        can_health[req->param1].can_data_speed = (bus_config[req->param1].can_data_speed / 10U);
        can_health[req->param1].canfd_enabled = bus_config[req->param1].canfd_enabled;
        can_health[req->param1].brs_enabled = bus_config[req->param1].brs_enabled;
        can_health[req->param1].canfd_non_iso = bus_config[req->param1].canfd_non_iso;
        resp_len = sizeof(can_health[req->param1]);
        (void)memcpy(resp, &can_health[req->param1], resp_len);
      }
      break;
    // **** 0xc3: fetch MCU UID
    case 0xc3:
      (void)memcpy(resp, ((uint8_t *)UID_BASE), 12);
      resp_len = 12;
      break;
    // **** 0xd0: fetch serial (aka the provisioned dongle ID)
    case 0xd0:
      // addresses are OTP
      if (req->param1 == 1U) {
        (void)memcpy(resp, (uint8_t *)DEVICE_SERIAL_NUMBER_ADDRESS, 0x10);
        resp_len = 0x10;
      } else {
        get_provision_chunk(resp);
        resp_len = PROVISION_CHUNK_LEN;
      }
      break;
    // **** 0xd1: enter bootloader mode
    case 0xd1:
      // this allows reflashing of the bootstub
      switch (req->param1) {
        case 0:
          // only allow bootloader entry on debug builds
          #ifdef ALLOW_DEBUG
            print("-> entering bootloader\n");
            enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
            NVIC_SystemReset();
          #endif
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
    // **** 0xd2: get health packet
    case 0xd2:
      resp_len = get_jungle_health_pkt(resp);
      break;
    // **** 0xd3: get first 64 bytes of signature
    case 0xd3:
      {
        resp_len = 64;
        char * code = (char*)_app_start;
        int code_len = _app_start[0];
        (void)memcpy(resp, &code[code_len], resp_len);
      }
      break;
    // **** 0xd4: get second 64 bytes of signature
    case 0xd4:
      {
        resp_len = 64;
        char * code = (char*)_app_start;
        int code_len = _app_start[0];
        (void)memcpy(resp, &code[code_len + 64], resp_len);
      }
      break;
    // **** 0xd6: get version
    case 0xd6:
      COMPILE_TIME_ASSERT(sizeof(gitversion) <= USBPACKET_MAX_SIZE);
      (void)memcpy(resp, gitversion, sizeof(gitversion));
      resp_len = sizeof(gitversion) - 1U;
      break;
    // **** 0xd8: reset ST
    case 0xd8:
      NVIC_SystemReset();
      break;
    // **** 0xdb: set OBD CAN multiplexing mode
    case 0xdb:
      if (req->param1 == 1U) {
        // Enable OBD CAN
        current_board->set_can_mode(CAN_MODE_OBD_CAN2);
      } else {
        // Disable OBD CAN
        current_board->set_can_mode(CAN_MODE_NORMAL);
      }
      break;
    // **** 0xdd: get healthpacket and CANPacket versions
    case 0xdd:
      resp[0] = JUNGLE_HEALTH_PACKET_VERSION;
      resp[1] = CAN_PACKET_VERSION;
      resp[2] = CAN_HEALTH_PACKET_VERSION;
      resp_len = 3;
      break;
    // **** 0xde: set can bitrate
    case 0xde:
      if ((req->param1 < PANDA_BUS_CNT) && is_speed_valid(req->param2, speeds, sizeof(speeds)/sizeof(speeds[0]))) {
        bus_config[req->param1].can_speed = req->param2;
        bool ret = can_init(CAN_NUM_FROM_BUS_NUM(req->param1));
        UNUSED(ret);
      }
      break;
    // **** 0xe0: debug read
    case 0xe0:
      // read
      while ((resp_len < MIN(req->length, USBPACKET_MAX_SIZE)) && get_char(get_ring_by_number(0), (char*)&resp[resp_len])) {
        ++resp_len;
      }
      break;
    // **** 0xe5: set CAN loopback (for testing)
    case 0xe5:
      can_loopback = (req->param1 > 0U);
      can_init_all();
      break;
    // **** 0xf1: Clear CAN ring buffer.
    case 0xf1:
      if (req->param1 == 0xFFFFU) {
        print("Clearing CAN Rx queue\n");
        can_clear(&can_rx_q);
      } else if (req->param1 < PANDA_BUS_CNT) {
        print("Clearing CAN Tx queue\n");
        can_clear(can_queues[req->param1]);
      } else {
        print("Clearing CAN CAN ring buffer failed: wrong bus number\n");
      }
      break;
    // **** 0xf2: Clear debug ring buffer.
    case 0xf2:
      print("Clearing debug queue.\n");
      clear_uart_buff(get_ring_by_number(0));
      break;
    // **** 0xf4: Set CAN transceiver enable pin
    case 0xf4:
      current_board->enable_can_transciever(req->param1, req->param2 > 0U);
      break;
    // **** 0xf5: Set CAN silent mode
    case 0xf5:
      can_silent = (req->param1 > 0U) ? ALL_CAN_SILENT : ALL_CAN_LIVE;
      can_init_all();
      break;
    // **** 0xf7: enable/disable header pin by number
    case 0xf7:
      current_board->enable_header_pin(req->param1, req->param2 > 0U);
      break;
    // **** 0xf9: set CAN FD data bitrate
    case 0xf9:
      if ((req->param1 < PANDA_CAN_CNT) &&
           current_board->has_canfd &&
           is_speed_valid(req->param2, data_speeds, sizeof(data_speeds)/sizeof(data_speeds[0]))) {
        bus_config[req->param1].can_data_speed = req->param2;
        bus_config[req->param1].canfd_enabled = (req->param2 >= bus_config[req->param1].can_speed);
        bus_config[req->param1].brs_enabled = (req->param2 > bus_config[req->param1].can_speed);
        bool ret = can_init(CAN_NUM_FROM_BUS_NUM(req->param1));
        UNUSED(ret);
      }
      break;
    // **** 0xfc: set CAN FD non-ISO mode
    case 0xfc:
      if ((req->param1 < PANDA_CAN_CNT) && current_board->has_canfd) {
        bus_config[req->param1].canfd_non_iso = (req->param2 != 0U);
        bool ret = can_init(CAN_NUM_FROM_BUS_NUM(req->param1));
        UNUSED(ret);
      }
      break;
    default:
      print("NO HANDLER ");
      puth(req->request);
      print("\n");
      break;
  }
  return resp_len;
}
