extern int _app_start[0xc000]; // Only first 3 sectors of size 0x4000 are used

// Prototypes
void set_safety_mode(uint16_t mode, uint16_t param);
bool is_car_safety_mode(uint16_t mode);

static int get_health_pkt(void *dat) {
  COMPILE_TIME_ASSERT(sizeof(struct health_t) <= USBPACKET_MAX_SIZE);
  struct health_t * health = (struct health_t*)dat;

  health->uptime_pkt = uptime_cnt;
  health->voltage_pkt = current_board->read_voltage_mV();
  health->current_pkt = current_board->read_current_mA();

  // Use the GPIO pin to determine ignition or use a CAN based logic
  health->ignition_line_pkt = (uint8_t)(current_board->check_ignition());
  health->ignition_can_pkt = ignition_can;

  health->controls_allowed_pkt = controls_allowed;
  health->safety_tx_blocked_pkt = safety_tx_blocked;
  health->safety_rx_invalid_pkt = safety_rx_invalid;
  health->tx_buffer_overflow_pkt = tx_buffer_overflow;
  health->rx_buffer_overflow_pkt = rx_buffer_overflow;
  health->car_harness_status_pkt = harness.status;
  health->safety_mode_pkt = (uint8_t)(current_safety_mode);
  health->safety_param_pkt = current_safety_param;
  health->alternative_experience_pkt = alternative_experience;
  health->power_save_enabled_pkt = power_save_status == POWER_SAVE_STATUS_ENABLED;
  health->heartbeat_lost_pkt = heartbeat_lost;
  health->safety_rx_checks_invalid_pkt = safety_rx_checks_invalid;

  health->spi_checksum_error_count_pkt = spi_checksum_error_count;

  health->fault_status_pkt = fault_status;
  health->faults_pkt = faults;

  health->interrupt_load_pkt = interrupt_load;

  health->fan_power = fan_state.power;
  health->fan_stall_count = fan_state.total_stall_count;

  health->sbu1_voltage_mV = harness.sbu1_voltage_mV;
  health->sbu2_voltage_mV = harness.sbu2_voltage_mV;

  health->som_reset_triggered = bootkick_reset_triggered;

  return sizeof(*health);
}

// send on serial, first byte to select the ring
void comms_endpoint2_write(const uint8_t *data, uint32_t len) {
  uart_ring *ur = get_ring_by_number(data[0]);
  if ((len != 0U) && (ur != NULL)) {
    if ((data[0] < 2U) || (data[0] >= 4U)) {
      for (uint32_t i = 1; i < len; i++) {
        while (!put_char(ur, data[i])) {
          // wait
        }
      }
    }
  }
}

int comms_control_handler(ControlPacket_t *req, uint8_t *resp) {
  unsigned int resp_len = 0;
  uart_ring *ur = NULL;
  uint32_t time;

#ifdef DEBUG_COMMS
  print("raw control request: "); hexdump(req, sizeof(ControlPacket_t)); print("\n");
  print("- request "); puth(req->request); print("\n");
  print("- param1 "); puth(req->param1); print("\n");
  print("- param2 "); puth(req->param2); print("\n");
#endif

  switch (req->request) {
    // **** 0xa8: get microsecond timer
    case 0xa8:
      time = microsecond_timer_get();
      resp[0] = (time & 0x000000FFU);
      resp[1] = ((time & 0x0000FF00U) >> 8U);
      resp[2] = ((time & 0x00FF0000U) >> 16U);
      resp[3] = ((time & 0xFF000000U) >> 24U);
      resp_len = 4U;
      break;
    // **** 0xb0: set IR power
    case 0xb0:
      current_board->set_ir_power(req->param1);
      break;
    // **** 0xb1: set fan power
    case 0xb1:
      fan_set_power(req->param1);
      break;
    // **** 0xb2: get fan rpm
    case 0xb2:
      resp[0] = (fan_state.rpm & 0x00FFU);
      resp[1] = ((fan_state.rpm & 0xFF00U) >> 8U);
      resp_len = 2;
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
        (void)memcpy(resp, (uint8_t*)(&can_health[req->param1]), resp_len);
      }
      break;
    // **** 0xc3: fetch MCU UID
    case 0xc3:
      (void)memcpy(resp, ((uint8_t *)UID_BASE), 12);
      resp_len = 12;
      break;
    // **** 0xc4: get interrupt call rate
    case 0xc4:
      if (req->param1 < NUM_INTERRUPTS) {
        uint32_t load = interrupts[req->param1].call_rate;
        resp[0] = (load & 0x000000FFU);
        resp[1] = ((load & 0x0000FF00U) >> 8U);
        resp[2] = ((load & 0x00FF0000U) >> 16U);
        resp[3] = ((load & 0xFF000000U) >> 24U);
        resp_len = 4U;
      }
      break;
    // **** 0xc5: DEBUG: drive relay
    case 0xc5:
      set_intercept_relay((req->param1 & 0x1U), (req->param1 & 0x2U));
      break;
    // **** 0xc6: DEBUG: read SOM GPIO
    case 0xc6:
      resp[0] = current_board->read_som_gpio();
      resp_len = 1;
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
      resp_len = get_health_pkt(resp);
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
      if (current_board->has_obd) {
        if (req->param1 == 1U) {
          // Enable OBD CAN
          current_board->set_can_mode(CAN_MODE_OBD_CAN2);
        } else {
          // Disable OBD CAN
          current_board->set_can_mode(CAN_MODE_NORMAL);
        }
      }
      break;

    // **** 0xdc: set safety mode
    case 0xdc:
      set_safety_mode(req->param1, (uint16_t)req->param2);
      break;
    // **** 0xdd: get healthpacket and CANPacket versions
    case 0xdd:
      resp[0] = HEALTH_PACKET_VERSION;
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
    // **** 0xdf: set alternative experience
    case 0xdf:
      // you can only set this if you are in a non car safety mode
      if (!is_car_safety_mode(current_safety_mode)) {
        alternative_experience = req->param1;
      }
      break;
    // **** 0xe0: uart read
    case 0xe0:
      ur = get_ring_by_number(req->param1);
      if (!ur) {
        break;
      }

      // read
      uint16_t req_length = MIN(req->length, USBPACKET_MAX_SIZE);
      while ((resp_len < req_length) &&
                         get_char(ur, (char*)&resp[resp_len])) {
        ++resp_len;
      }
      break;
    // **** 0xe1: uart set baud rate
    case 0xe1:
      ur = get_ring_by_number(req->param1);
      if (!ur) {
        break;
      }
      uart_set_baud(ur->uart, req->param2);
      break;
    // **** 0xe2: uart set parity
    case 0xe2:
      ur = get_ring_by_number(req->param1);
      if (!ur) {
        break;
      }
      switch (req->param2) {
        case 0:
          // disable parity, 8-bit
          ur->uart->CR1 &= ~(USART_CR1_PCE | USART_CR1_M);
          break;
        case 1:
          // even parity, 9-bit
          ur->uart->CR1 &= ~USART_CR1_PS;
          ur->uart->CR1 |= USART_CR1_PCE | USART_CR1_M;
          break;
        case 2:
          // odd parity, 9-bit
          ur->uart->CR1 |= USART_CR1_PS;
          ur->uart->CR1 |= USART_CR1_PCE | USART_CR1_M;
          break;
        default:
          break;
      }
      break;
    // **** 0xe4: uart set baud rate extended
    case 0xe4:
      ur = get_ring_by_number(req->param1);
      if (!ur) {
        break;
      }
      uart_set_baud(ur->uart, (int)req->param2*300);
      break;
    // **** 0xe5: set CAN loopback (for testing)
    case 0xe5:
      can_loopback = req->param1 > 0U;
      can_init_all();
      break;
    // **** 0xe6: set custom clock source period
    case 0xe6:
      clock_source_set_period(req->param1);
      break;
    // **** 0xe7: set power save state
    case 0xe7:
      set_power_save_state(req->param1);
      break;
    // **** 0xe8: set can-fd auto swithing mode
    case 0xe8:
      bus_config[req->param1].canfd_auto = req->param2 > 0U;
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
    // **** 0xf2: Clear UART ring buffer.
    case 0xf2:
      {
        uart_ring * rb = get_ring_by_number(req->param1);
        if (rb != NULL) {
          print("Clearing UART queue.\n");
          clear_uart_buff(rb);
        }
        break;
      }
    // **** 0xf3: Heartbeat. Resets heartbeat counter.
    case 0xf3:
      {
        heartbeat_counter = 0U;
        heartbeat_lost = false;
        heartbeat_disabled = false;
        heartbeat_engaged = (req->param1 == 1U);
        break;
      }
    // **** 0xf6: set siren enabled
    case 0xf6:
      siren_enabled = (req->param1 != 0U);
      break;
    // **** 0xf7: set green led enabled
    case 0xf7:
      green_led_enabled = (req->param1 != 0U);
      break;
    // **** 0xf8: disable heartbeat checks
    case 0xf8:
      if (!is_car_safety_mode(current_safety_mode)) {
        heartbeat_disabled = true;
      }
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
