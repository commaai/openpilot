// ********************* Includes *********************
#include "config.h"

#include "drivers/pwm.h"
#include "drivers/usb.h"
#include "drivers/gmlan_alt.h"
#include "drivers/kline_init.h"

#include "early_init.h"
#include "provision.h"

#include "power_saving.h"
#include "safety.h"

#include "drivers/can_common.h"

#ifdef STM32H7
  #include "drivers/fdcan.h"
#else
  #include "drivers/bxcan.h"
#endif

#include "usb_protocol.h"

#include "obj/gitversion.h"

extern int _app_start[0xc000]; // Only first 3 sectors of size 0x4000 are used

// When changing this struct, boardd and python/__init__.py needs to be kept up to date!
#define HEALTH_PACKET_VERSION 1
struct __attribute__((packed)) health_t {
  uint32_t uptime_pkt;
  uint32_t voltage_pkt;
  uint32_t current_pkt;
  uint32_t can_rx_errs_pkt;
  uint32_t can_send_errs_pkt;
  uint32_t can_fwd_errs_pkt;
  uint32_t gmlan_send_errs_pkt;
  uint32_t faults_pkt;
  uint8_t ignition_line_pkt;
  uint8_t ignition_can_pkt;
  uint8_t controls_allowed_pkt;
  uint8_t gas_interceptor_detected_pkt;
  uint8_t car_harness_status_pkt;
  uint8_t usb_power_mode_pkt;
  uint8_t safety_mode_pkt;
  int16_t safety_param_pkt;
  uint8_t fault_status_pkt;
  uint8_t power_save_enabled_pkt;
  uint8_t heartbeat_lost_pkt;
};


// ********************* Serial debugging *********************

bool check_started(void) {
  return current_board->check_ignition() || ignition_can;
}

void debug_ring_callback(uart_ring *ring) {
  char rcv;
  while (getc(ring, &rcv)) {
    (void)putc(ring, rcv);  // misra-c2012-17.7: cast to void is ok: debug function

    // only allow bootloader entry on debug builds
    #ifdef ALLOW_DEBUG
      // jump to DFU flash
      if (rcv == 'z') {
        enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
        NVIC_SystemReset();
      }
    #endif

    // normal reset
    if (rcv == 'x') {
      NVIC_SystemReset();
    }

    // enable CDP mode
    if (rcv == 'C') {
      puts("switching USB to CDP mode\n");
      current_board->set_usb_power_mode(USB_POWER_CDP);
    }
    if (rcv == 'c') {
      puts("switching USB to client mode\n");
      current_board->set_usb_power_mode(USB_POWER_CLIENT);
    }
    if (rcv == 'D') {
      puts("switching USB to DCP mode\n");
      current_board->set_usb_power_mode(USB_POWER_DCP);
    }
  }
}

// ****************************** safety mode ******************************

// this is the only way to leave silent mode
void set_safety_mode(uint16_t mode, int16_t param) {
  uint16_t mode_copy = mode;
  int err = set_safety_hooks(mode_copy, param);
  if (err == -1) {
    puts("Error: safety set mode failed. Falling back to SILENT\n");
    mode_copy = SAFETY_SILENT;
    err = set_safety_hooks(mode_copy, 0);
    if (err == -1) {
      puts("Error: Failed setting SILENT mode. Hanging\n");
      while (true) {
        // TERMINAL ERROR: we can't continue if SILENT safety mode isn't succesfully set
      }
    }
  }
  switch (mode_copy) {
    case SAFETY_SILENT:
      set_intercept_relay(false);
      if (current_board->has_obd) {
        current_board->set_can_mode(CAN_MODE_NORMAL);
      }
      can_silent = ALL_CAN_SILENT;
      break;
    case SAFETY_NOOUTPUT:
      set_intercept_relay(false);
      if (current_board->has_obd) {
        current_board->set_can_mode(CAN_MODE_NORMAL);
      }
      can_silent = ALL_CAN_LIVE;
      break;
    case SAFETY_ELM327:
      set_intercept_relay(false);
      heartbeat_counter = 0U;
      heartbeat_lost = false;
      if (current_board->has_obd) {
        if (param == 0) {
          current_board->set_can_mode(CAN_MODE_OBD_CAN2);
        } else {
          current_board->set_can_mode(CAN_MODE_NORMAL);
        }
      }
      can_silent = ALL_CAN_LIVE;
      break;
    default:
      set_intercept_relay(true);
      heartbeat_counter = 0U;
      heartbeat_lost = false;
      if (current_board->has_obd) {
        current_board->set_can_mode(CAN_MODE_NORMAL);
      }
      can_silent = ALL_CAN_LIVE;
      break;
  }
  can_init_all();
}

bool is_car_safety_mode(uint16_t mode) {
  return (mode != SAFETY_SILENT) &&
         (mode != SAFETY_NOOUTPUT) &&
         (mode != SAFETY_ELM327);
}

// ***************************** USB port *****************************

int get_health_pkt(void *dat) {
  COMPILE_TIME_ASSERT(sizeof(struct health_t) <= MAX_RESP_LEN);
  struct health_t * health = (struct health_t*)dat;

  health->uptime_pkt = uptime_cnt;
  health->voltage_pkt = adc_get_voltage();
  health->current_pkt = current_board->read_current();

  //Use the GPIO pin to determine ignition or use a CAN based logic
  health->ignition_line_pkt = (uint8_t)(current_board->check_ignition());
  health->ignition_can_pkt = (uint8_t)(ignition_can);

  health->controls_allowed_pkt = controls_allowed;
  health->gas_interceptor_detected_pkt = gas_interceptor_detected;
  health->can_rx_errs_pkt = can_rx_errs;
  health->can_send_errs_pkt = can_send_errs;
  health->can_fwd_errs_pkt = can_fwd_errs;
  health->gmlan_send_errs_pkt = gmlan_send_errs;
  health->car_harness_status_pkt = car_harness_status;
  health->usb_power_mode_pkt = usb_power_mode;
  health->safety_mode_pkt = (uint8_t)(current_safety_mode);
  health->safety_param_pkt = current_safety_param;
  health->power_save_enabled_pkt = (uint8_t)(power_save_status == POWER_SAVE_STATUS_ENABLED);
  health->heartbeat_lost_pkt = (uint8_t)(heartbeat_lost);

  health->fault_status_pkt = fault_status;
  health->faults_pkt = faults;

  return sizeof(*health);
}

int get_rtc_pkt(void *dat) {
  timestamp_t t = rtc_get_time();
  (void)memcpy(dat, &t, sizeof(t));
  return sizeof(t);
}



// send on serial, first byte to select the ring
void usb_cb_ep2_out(void *usbdata, int len, bool hardwired) {
  UNUSED(hardwired);
  uint8_t *usbdata8 = (uint8_t *)usbdata;
  uart_ring *ur = get_ring_by_number(usbdata8[0]);
  if ((len != 0) && (ur != NULL)) {
    if ((usbdata8[0] < 2U) || safety_tx_lin_hook(usbdata8[0] - 2U, &usbdata8[1], len - 1)) {
      for (int i = 1; i < len; i++) {
        while (!putc(ur, usbdata8[i])) {
          // wait
        }
      }
    }
  }
}

void usb_cb_ep3_out_complete(void) {
  if (can_tx_check_min_slots_free(MAX_CAN_MSGS_PER_BULK_TRANSFER)) {
    usb_outep3_resume_if_paused();
  }
}

void usb_cb_enumeration_complete(void) {
  puts("USB enumeration complete\n");
  is_enumerated = 1;
}

int usb_cb_control_msg(USB_Setup_TypeDef *setup, uint8_t *resp, bool hardwired) {
  unsigned int resp_len = 0;
  uart_ring *ur = NULL;
  timestamp_t t;
  switch (setup->b.bRequest) {
    // **** 0xa0: get rtc time
    case 0xa0:
      resp_len = get_rtc_pkt(resp);
      break;
    // **** 0xa1: set rtc year
    case 0xa1:
      t = rtc_get_time();
      t.year = setup->b.wValue.w;
      rtc_set_time(t);
      break;
    // **** 0xa2: set rtc month
    case 0xa2:
      t = rtc_get_time();
      t.month = setup->b.wValue.w;
      rtc_set_time(t);
      break;
    // **** 0xa3: set rtc day
    case 0xa3:
      t = rtc_get_time();
      t.day = setup->b.wValue.w;
      rtc_set_time(t);
      break;
    // **** 0xa4: set rtc weekday
    case 0xa4:
      t = rtc_get_time();
      t.weekday = setup->b.wValue.w;
      rtc_set_time(t);
      break;
    // **** 0xa5: set rtc hour
    case 0xa5:
      t = rtc_get_time();
      t.hour = setup->b.wValue.w;
      rtc_set_time(t);
      break;
    // **** 0xa6: set rtc minute
    case 0xa6:
      t = rtc_get_time();
      t.minute = setup->b.wValue.w;
      rtc_set_time(t);
      break;
    // **** 0xa7: set rtc second
    case 0xa7:
      t = rtc_get_time();
      t.second = setup->b.wValue.w;
      rtc_set_time(t);
      break;
    // **** 0xb0: set IR power
    case 0xb0:
      current_board->set_ir_power(setup->b.wValue.w);
      break;
    // **** 0xb1: set fan power
    case 0xb1:
      current_board->set_fan_power(setup->b.wValue.w);
      break;
    // **** 0xb2: get fan rpm
    case 0xb2:
      resp[0] = (fan_rpm & 0x00FFU);
      resp[1] = ((fan_rpm & 0xFF00U) >> 8U);
      resp_len = 2;
      break;
    // **** 0xb3: set phone power
    case 0xb3:
      current_board->set_phone_power(setup->b.wValue.w > 0U);
      break;
    // **** 0xc0: get CAN debug info
    case 0xc0:
      puts("can tx: "); puth(can_tx_cnt);
      puts(" txd: "); puth(can_txd_cnt);
      puts(" rx: "); puth(can_rx_cnt);
      puts(" err: "); puth(can_err_cnt);
      puts("\n");
      break;
    // **** 0xc1: get hardware type
    case 0xc1:
      resp[0] = hw_type;
      resp_len = 1;
      break;
    // **** 0xd0: fetch serial number
    case 0xd0:
      // addresses are OTP
      if (setup->b.wValue.w == 1U) {
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
      // so it's blocked over wifi
      switch (setup->b.wValue.w) {
        case 0:
          // only allow bootloader entry on debug builds
          #ifdef ALLOW_DEBUG
            if (hardwired) {
              puts("-> entering bootloader\n");
              enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
              NVIC_SystemReset();
            }
          #endif
          break;
        case 1:
          puts("-> entering softloader\n");
          enter_bootloader_mode = ENTER_SOFTLOADER_MAGIC;
          NVIC_SystemReset();
          break;
        default:
          puts("Bootloader mode invalid\n");
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
      COMPILE_TIME_ASSERT(sizeof(gitversion) <= MAX_RESP_LEN);
      (void)memcpy(resp, gitversion, sizeof(gitversion));
      resp_len = sizeof(gitversion) - 1U;
      break;
    // **** 0xd8: reset ST
    case 0xd8:
      NVIC_SystemReset();
      break;
    // **** 0xd9: set ESP power
    case 0xd9:
      if (setup->b.wValue.w == 1U) {
        current_board->set_gps_mode(GPS_ENABLED);
      } else if (setup->b.wValue.w == 2U) {
        current_board->set_gps_mode(GPS_BOOTMODE);
      } else {
        current_board->set_gps_mode(GPS_DISABLED);
      }
      break;
    // **** 0xda: reset ESP, with optional boot mode
    case 0xda:
      current_board->set_gps_mode(GPS_DISABLED);
      delay(1000000);
      if (setup->b.wValue.w == 1U) {
        current_board->set_gps_mode(GPS_BOOTMODE);
      } else {
        current_board->set_gps_mode(GPS_ENABLED);
      }
      delay(1000000);
      current_board->set_gps_mode(GPS_ENABLED);
      break;
    // **** 0xdb: set GMLAN (white/grey) or OBD CAN (black) multiplexing mode
    case 0xdb:
      if(current_board->has_obd){
        if (setup->b.wValue.w == 1U) {
          // Enable OBD CAN
          current_board->set_can_mode(CAN_MODE_OBD_CAN2);
        } else {
          // Disable OBD CAN
          current_board->set_can_mode(CAN_MODE_NORMAL);
        }
      } else {
        if (setup->b.wValue.w == 1U) {
          // GMLAN ON
          if (setup->b.wIndex.w == 1U) {
            can_set_gmlan(1);
          } else if (setup->b.wIndex.w == 2U) {
            can_set_gmlan(2);
          } else {
            puts("Invalid bus num for GMLAN CAN set\n");
          }
        } else {
          can_set_gmlan(-1);
        }
      }
      break;

    // **** 0xdc: set safety mode
    case 0xdc:
      // Blocked over WiFi.
      // Allow SILENT, NOOUTPUT and ELM security mode to be set over wifi.
      if (hardwired || (setup->b.wValue.w == SAFETY_SILENT) ||
                       (setup->b.wValue.w == SAFETY_NOOUTPUT) ||
                       (setup->b.wValue.w == SAFETY_ELM327)) {
        set_safety_mode(setup->b.wValue.w, (uint16_t) setup->b.wIndex.w);
      }
      break;
    // **** 0xdd: get healthpacket and CANPacket versions
    case 0xdd:
      resp[0] = HEALTH_PACKET_VERSION;
      resp[1] = CAN_PACKET_VERSION;
      resp_len = 2;
      break;
    // **** 0xde: set can bitrate
    case 0xde:
      if (setup->b.wValue.w < BUS_MAX) {
        // TODO: add sanity check, ideally check if value is correct(from array of correct values)
        bus_config[setup->b.wValue.w].can_speed = setup->b.wIndex.w;
        bool ret = can_init(CAN_NUM_FROM_BUS_NUM(setup->b.wValue.w));
        UNUSED(ret);
      }
      break;
    // **** 0xdf: set unsafe mode
    case 0xdf:
      // you can only set this if you are in a non car safety mode
      if (!is_car_safety_mode(current_safety_mode)) {
        unsafe_mode = setup->b.wValue.w;
      }
      break;
    // **** 0xe0: uart read
    case 0xe0:
      ur = get_ring_by_number(setup->b.wValue.w);
      if (!ur) {
        break;
      }

      // TODO: Remove this again and fix boardd code to hande the message bursts instead of single chars
      if (ur == &uart_ring_gps) {
        dma_pointer_handler(ur, DMA2_Stream5->NDTR);
      }

      // read
      while ((resp_len < MIN(setup->b.wLength.w, MAX_RESP_LEN)) &&
                         getc(ur, (char*)&resp[resp_len])) {
        ++resp_len;
      }
      break;
    // **** 0xe1: uart set baud rate
    case 0xe1:
      ur = get_ring_by_number(setup->b.wValue.w);
      if (!ur) {
        break;
      }
      uart_set_baud(ur->uart, setup->b.wIndex.w);
      break;
    // **** 0xe2: uart set parity
    case 0xe2:
      ur = get_ring_by_number(setup->b.wValue.w);
      if (!ur) {
        break;
      }
      switch (setup->b.wIndex.w) {
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
      ur = get_ring_by_number(setup->b.wValue.w);
      if (!ur) {
        break;
      }
      uart_set_baud(ur->uart, (int)setup->b.wIndex.w*300);
      break;
    // **** 0xe5: set CAN loopback (for testing)
    case 0xe5:
      can_loopback = (setup->b.wValue.w > 0U);
      can_init_all();
      break;
    // **** 0xe6: set USB power
    case 0xe6:
      current_board->set_usb_power_mode(setup->b.wValue.w);
      break;
    // **** 0xe7: set power save state
    case 0xe7:
      set_power_save_state(setup->b.wValue.w);
      break;
    // **** 0xf0: k-line/l-line wake-up pulse for KWP2000 fast initialization
    case 0xf0:
      if(current_board->has_lin) {
        bool k = (setup->b.wValue.w == 0U) || (setup->b.wValue.w == 2U);
        bool l = (setup->b.wValue.w == 1U) || (setup->b.wValue.w == 2U);
        if (bitbang_wakeup(k, l)) {
          resp_len = -1; // do not clear NAK yet (wait for bit banging to finish)
        }
      }
      break;
    // **** 0xf1: Clear CAN ring buffer.
    case 0xf1:
      if (setup->b.wValue.w == 0xFFFFU) {
        puts("Clearing CAN Rx queue\n");
        can_clear(&can_rx_q);
      } else if (setup->b.wValue.w < BUS_MAX) {
        puts("Clearing CAN Tx queue\n");
        can_clear(can_queues[setup->b.wValue.w]);
      } else {
        puts("Clearing CAN CAN ring buffer failed: wrong bus number\n");
      }
      break;
    // **** 0xf2: Clear UART ring buffer.
    case 0xf2:
      {
        uart_ring * rb = get_ring_by_number(setup->b.wValue.w);
        if (rb != NULL) {
          puts("Clearing UART queue.\n");
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
        break;
      }
    // **** 0xf4: k-line/l-line 5 baud initialization
    case 0xf4:
      if(current_board->has_lin) {
        bool k = (setup->b.wValue.w == 0U) || (setup->b.wValue.w == 2U);
        bool l = (setup->b.wValue.w == 1U) || (setup->b.wValue.w == 2U);
        uint8_t five_baud_addr = (setup->b.wIndex.w & 0xFFU);
        if (bitbang_five_baud_addr(k, l, five_baud_addr)) {
          resp_len = -1; // do not clear NAK yet (wait for bit banging to finish)
        }
      }
      break;
    // **** 0xf5: set clock source mode
    case 0xf5:
      current_board->set_clock_source_mode(setup->b.wValue.w);
      break;
    // **** 0xf6: set siren enabled
    case 0xf6:
      siren_enabled = (setup->b.wValue.w != 0U);
      break;
    // **** 0xf7: set green led enabled
    case 0xf7:
      green_led_enabled = (setup->b.wValue.w != 0U);
      break;
#ifdef ALLOW_DEBUG
    // **** 0xf8: disable heartbeat checks
    case 0xf8:
      heartbeat_disabled = true;
      break;
#endif
    // **** 0xde: set CAN FD data bitrate
    case 0xf9:
      if (setup->b.wValue.w < CAN_MAX) {
        // TODO: add sanity check, ideally check if value is correct(from array of correct values)
        bus_config[setup->b.wValue.w].can_data_speed = setup->b.wIndex.w;
        bus_config[setup->b.wValue.w].canfd_enabled = (setup->b.wIndex.w >= bus_config[setup->b.wValue.w].can_speed) ? true : false;
        bus_config[setup->b.wValue.w].brs_enabled = (setup->b.wIndex.w > bus_config[setup->b.wValue.w].can_speed) ? true : false;
        bool ret = can_init(CAN_NUM_FROM_BUS_NUM(setup->b.wValue.w));
        UNUSED(ret);
      }
      break;
    // **** 0xfa: check if CAN FD and BRS are enabled
    case 0xfa:
      if (setup->b.wValue.w < CAN_MAX) {
        resp[0] =  bus_config[setup->b.wValue.w].canfd_enabled;
        resp[1] = bus_config[setup->b.wValue.w].brs_enabled;
        resp_len = 2;
      }
      break;
    default:
      puts("NO HANDLER ");
      puth(setup->b.bRequest);
      puts("\n");
      break;
  }
  return resp_len;
}

// ***************************** main code *****************************

// cppcheck-suppress unusedFunction ; used in headers not included in cppcheck
void __initialize_hardware_early(void) {
  early_initialization();
}

void __attribute__ ((noinline)) enable_fpu(void) {
  // enable the FPU
  SCB->CPACR |= ((3UL << (10U * 2U)) | (3UL << (11U * 2U)));
}

// go into SILENT when heartbeat isn't received for this amount of seconds.
#define HEARTBEAT_IGNITION_CNT_ON 5U
#define HEARTBEAT_IGNITION_CNT_OFF 2U

// called at 8Hz
uint8_t loop_counter = 0U;
void tick_handler(void) {
  if (TICK_TIMER->SR != 0) {
    // siren
    current_board->set_siren((loop_counter & 1U) && (siren_enabled || (siren_countdown > 0U)));

    // decimated to 1Hz
    if (loop_counter == 0U) {
      can_live = pending_can_live;

      current_board->usb_power_mode_tick(uptime_cnt);

      //puth(usart1_dma); puts(" "); puth(DMA2_Stream5->M0AR); puts(" "); puth(DMA2_Stream5->NDTR); puts("\n");

      // reset this every 16th pass
      if ((uptime_cnt & 0xFU) == 0U) {
        pending_can_live = 0;
      }
      #ifdef DEBUG
        puts("** blink ");
        puts("rx:"); puth4(can_rx_q.r_ptr); puts("-"); puth4(can_rx_q.w_ptr); puts("  ");
        puts("tx1:"); puth4(can_tx1_q.r_ptr); puts("-"); puth4(can_tx1_q.w_ptr); puts("  ");
        puts("tx2:"); puth4(can_tx2_q.r_ptr); puts("-"); puth4(can_tx2_q.w_ptr); puts("  ");
        puts("tx3:"); puth4(can_tx3_q.r_ptr); puts("-"); puth4(can_tx3_q.w_ptr); puts("\n");
      #endif

      // Tick drivers
      fan_tick();

      // set green LED to be controls allowed
      current_board->set_led(LED_GREEN, controls_allowed | green_led_enabled);

      // turn off the blue LED, turned on by CAN
      // unless we are in power saving mode
      current_board->set_led(LED_BLUE, (uptime_cnt & 1U) && (power_save_status == POWER_SAVE_STATUS_ENABLED));

      // increase heartbeat counter and cap it at the uint32 limit
      if (heartbeat_counter < __UINT32_MAX__) {
        heartbeat_counter += 1U;
      }

      if (siren_countdown > 0U) {
        siren_countdown -= 1U;
      }

      if (controls_allowed) {
        controls_allowed_countdown = 30U;
      } else if (controls_allowed_countdown > 0U) {
        controls_allowed_countdown -= 1U;
      } else {

      }

      if (!heartbeat_disabled) {
        // if the heartbeat has been gone for a while, go to SILENT safety mode and enter power save
        if (heartbeat_counter >= (check_started() ? HEARTBEAT_IGNITION_CNT_ON : HEARTBEAT_IGNITION_CNT_OFF)) {
          puts("device hasn't sent a heartbeat for 0x");
          puth(heartbeat_counter);
          puts(" seconds. Safety is set to SILENT mode.\n");

          if (controls_allowed_countdown > 0U) {
            siren_countdown = 5U;
            controls_allowed_countdown = 0U;
          }

          // set flag to indicate the heartbeat was lost
          if (is_car_safety_mode(current_safety_mode)) {
            heartbeat_lost = true;
          }

          if (current_safety_mode != SAFETY_SILENT) {
            set_safety_mode(SAFETY_SILENT, 0U);
          }
          if (power_save_status != POWER_SAVE_STATUS_ENABLED) {
            set_power_save_state(POWER_SAVE_STATUS_ENABLED);
          }

          // Also disable IR when the heartbeat goes missing
          current_board->set_ir_power(0U);

          // If enumerated but no heartbeat (phone up, boardd not running), turn the fan on to cool the device
          if(usb_enumerated()){
            current_board->set_fan_power(50U);
          } else {
            current_board->set_fan_power(0U);
          }
        }

        // enter CDP mode when car starts to ensure we are charging a turned off EON
        if (check_started() && (usb_power_mode != USB_POWER_CDP)) {
          current_board->set_usb_power_mode(USB_POWER_CDP);
        }
      }

      // check registers
      check_registers();

      // set ignition_can to false after 2s of no CAN seen
      if (ignition_can_cnt > 2U) {
        ignition_can = false;
      }

      // on to the next one
      uptime_cnt += 1U;
      safety_mode_cnt += 1U;
      ignition_can_cnt += 1U;

      // synchronous safety check
      safety_tick(current_rx_checks);
    }

    loop_counter++;
    loop_counter %= 8U;
  }
  TICK_TIMER->SR = 0;
}


int main(void) {
  // Init interrupt table
  init_interrupts(true);

  // shouldn't have interrupts here, but just in case
  disable_interrupts();

  // init early devices
  clock_init();
  peripherals_init();
  detect_external_debug_serial();
  detect_board_type();
  adc_init();

  // print hello
  puts("\n\n\n************************ MAIN START ************************\n");

  // check for non-supported board types
  if(hw_type == HW_TYPE_UNKNOWN){
    puts("Unsupported board type\n");
    while (1) { /* hang */ }
  }

  puts("Config:\n");
  puts("  Board type: "); puts(current_board->board_type); puts("\n");
  puts(has_external_debug_serial ? "  Real serial\n" : "  USB serial\n");

  // init board
  current_board->init();

  // panda has an FPU, let's use it!
  enable_fpu();

  // enable main uart if it's connected
  if (has_external_debug_serial) {
    // WEIRDNESS: without this gate around the UART, it would "crash", but only if the ESP is enabled
    // assuming it's because the lines were left floating and spurious noise was on them
    uart_init(&uart_ring_debug, 115200);
  }

  if (current_board->has_gps) {
    uart_init(&uart_ring_gps, 9600);
  } else {
    // enable ESP uart
    uart_init(&uart_ring_gps, 115200);
  }

  if(current_board->has_lin){
    // enable LIN
    uart_init(&uart_ring_lin1, 10400);
    UART5->CR2 |= USART_CR2_LINEN;
    uart_init(&uart_ring_lin2, 10400);
    USART3->CR2 |= USART_CR2_LINEN;
  }

  microsecond_timer_init();

  // init to SILENT and can silent
  set_safety_mode(SAFETY_SILENT, 0);

  // enable CAN TXs
  current_board->enable_can_transceivers(true);

  // 8Hz timer
  REGISTER_INTERRUPT(TICK_TIMER_IRQ, tick_handler, 10U, FAULT_INTERRUPT_RATE_TICK)
  tick_timer_init();

#ifdef DEBUG
  puts("DEBUG ENABLED\n");
#endif
  // enable USB (right before interrupts or enum can fail!)
  usb_init();

  puts("**** INTERRUPTS ON ****\n");
  enable_interrupts();

  // LED should keep on blinking all the time
  uint64_t cnt = 0;

  for (cnt=0;;cnt++) {
    if (power_save_status == POWER_SAVE_STATUS_DISABLED) {
      #ifdef DEBUG_FAULTS
      if(fault_status == FAULT_STATUS_NONE){
      #endif
        uint32_t div_mode = ((usb_power_mode == USB_POWER_DCP) ? 4U : 1U);

        // useful for debugging, fade breaks = panda is overloaded
        for(uint32_t fade = 0U; fade < MAX_LED_FADE; fade += div_mode){
          current_board->set_led(LED_RED, true);
          delay(fade >> 4);
          current_board->set_led(LED_RED, false);
          delay((MAX_LED_FADE - fade) >> 4);
        }

        for(uint32_t fade = MAX_LED_FADE; fade > 0U; fade -= div_mode){
          current_board->set_led(LED_RED, true);
          delay(fade >> 4);
          current_board->set_led(LED_RED, false);
          delay((MAX_LED_FADE - fade) >> 4);
        }

      #ifdef DEBUG_FAULTS
      } else {
          current_board->set_led(LED_RED, 1);
          delay(512000U);
          current_board->set_led(LED_RED, 0);
          delay(512000U);
        }
      #endif
    } else {
      __WFI();
    }
  }

  return 0;
}
