#define GMLAN_TICKS_PER_SECOND 33300 //1sec @ 33.3kbps
#define GMLAN_TICKS_PER_TIMEOUT_TICKLE 500 //15ms @ 33.3kbps
#define GMLAN_HIGH 0 //0 is high on bus (dominant)
#define GMLAN_LOW 1 //1 is low on bus

#define DISABLED -1
#define BITBANG 0
#define GPIO_SWITCH 1

#define MAX_BITS_CAN_PACKET (200)

int gmlan_alt_mode = DISABLED;

// returns out_len
int do_bitstuff(char *out, char *in, int in_len) {
  int last_bit = -1;
  int bit_cnt = 0;
  int j = 0;
  for (int i = 0; i < in_len; i++) {
    char bit = in[i];
    out[j] = bit;
    j++;

    // do the stuffing
    if (bit == last_bit) {
      bit_cnt++;
      if (bit_cnt == 5) {
        // 5 in a row the same, do stuff
        last_bit = !bit;
        out[j] = last_bit;
        j++;
        bit_cnt = 1;
      }
    } else {
      // this is a new bit
      last_bit = bit;
      bit_cnt = 1;
    }
  }
  return j;
}

int append_crc(char *in, int in_len) {
  unsigned int crc = 0;
  for (int i = 0; i < in_len; i++) {
    crc <<= 1;
    if (((unsigned int)(in[i]) ^ ((crc >> 15) & 1U)) != 0U) {
      crc = crc ^ 0x4599U;
    }
    crc &= 0x7fffU;
  }
  int in_len_copy = in_len;
  for (int i = 14; i >= 0; i--) {
    in[in_len_copy] = (crc >> (unsigned int)(i)) & 1U;
    in_len_copy++;
  }
  return in_len_copy;
}

int append_bits(char *in, int in_len, char *app, int app_len) {
  int in_len_copy = in_len;
  for (int i = 0; i < app_len; i++) {
    in[in_len_copy] = app[i];
    in_len_copy++;
  }
  return in_len_copy;
}

int append_int(char *in, int in_len, int val, int val_len) {
  int in_len_copy = in_len;
  for (int i = val_len - 1; i >= 0; i--) {
    in[in_len_copy] = ((unsigned int)(val) & (1U << (unsigned int)(i))) != 0U;
    in_len_copy++;
  }
  return in_len_copy;
}

int get_bit_message(char *out, CANPacket_t *to_bang) {
  char pkt[MAX_BITS_CAN_PACKET];
  char footer[] = {
    1,  // CRC delimiter
    1,  // ACK
    1,  // ACK delimiter
    1,1,1,1,1,1,1, // EOF
    1,1,1, // IFS
  };

  int len = 0;

  // test packet
  int dlc_len = GET_LEN(to_bang);
  len = append_int(pkt, len, 0, 1);    // Start-of-frame

  if (to_bang->extended != 0U) {
    // extended identifier
    len = append_int(pkt, len, GET_ADDR(to_bang) >> 18, 11);  // Identifier
    len = append_int(pkt, len, 3, 2);    // SRR+IDE
    len = append_int(pkt, len, (GET_ADDR(to_bang)) & ((1U << 18) - 1U), 18);  // Identifier
    len = append_int(pkt, len, 0, 3);    // RTR+r1+r0
  } else {
    // standard identifier
    len = append_int(pkt, len, GET_ADDR(to_bang), 11);  // Identifier
    len = append_int(pkt, len, 0, 3);    // RTR+IDE+reserved
  }

  len = append_int(pkt, len, dlc_len, 4);    // Data length code

  // append data
  for (int i = 0; i < dlc_len; i++) {
    len = append_int(pkt, len, to_bang->data[i], 8);
  }

  // append crc
  len = append_crc(pkt, len);

  // do bitstuffing
  len = do_bitstuff(out, pkt, len);

  // append footer
  len = append_bits(out, len, footer, sizeof(footer));
  return len;
}

void TIM12_IRQ_Handler(void);

void setup_timer(void) {
  // register interrupt
  REGISTER_INTERRUPT(TIM8_BRK_TIM12_IRQn, TIM12_IRQ_Handler, 40000U, FAULT_INTERRUPT_RATE_GMLAN)

  // setup
  register_set(&(TIM12->PSC), (APB1_TIMER_FREQ-1U), 0xFFFFU);    // Tick on 1 us
  register_set(&(TIM12->CR1), TIM_CR1_CEN, 0x3FU); // Enable
  register_set(&(TIM12->ARR), (30U-1U), 0xFFFFU);   // 33.3 kbps

  // in case it's disabled
  NVIC_EnableIRQ(TIM8_BRK_TIM12_IRQn);

  // run the interrupt
  register_set(&(TIM12->DIER), TIM_DIER_UIE, 0x5F5FU); // Update interrupt
  TIM12->SR = 0;
}

int gmlan_timeout_counter = GMLAN_TICKS_PER_TIMEOUT_TICKLE; //GMLAN transceiver times out every 17ms held high; tickle every 15ms
int can_timeout_counter = GMLAN_TICKS_PER_SECOND; //1 second

int inverted_bit_to_send = GMLAN_HIGH;
int gmlan_switch_below_timeout = -1;
int gmlan_switch_timeout_enable = 0;

void gmlan_switch_init(int timeout_enable) {
  gmlan_switch_timeout_enable = timeout_enable;
  gmlan_alt_mode = GPIO_SWITCH;
  gmlan_switch_below_timeout = 1;
  set_gpio_mode(GPIOB, 13, MODE_OUTPUT);

  setup_timer();

  inverted_bit_to_send = GMLAN_LOW; //We got initialized, set the output low
}

void set_gmlan_digital_output(int to_set) {
  inverted_bit_to_send = to_set;
  /*
  print("Writing ");
  puth(inverted_bit_to_send);
  print("\n");
  */
}

void reset_gmlan_switch_timeout(void) {
  can_timeout_counter = GMLAN_TICKS_PER_SECOND;
  gmlan_switch_below_timeout = 1;
  gmlan_alt_mode = GPIO_SWITCH;
}

void set_bitbanged_gmlan(int val) {
  if (val != 0) {
    register_set_bits(&(GPIOB->ODR), (1U << 13));
  } else {
    register_clear_bits(&(GPIOB->ODR), (1U << 13));
  }
}

char pkt_stuffed[MAX_BITS_CAN_PACKET];
int gmlan_sending = -1;
int gmlan_sendmax = -1;
bool gmlan_send_ok = true;

int gmlan_silent_count = 0;
int gmlan_fail_count = 0;
#define REQUIRED_SILENT_TIME 10
#define MAX_FAIL_COUNT 10

void TIM12_IRQ_Handler(void) {
  if (gmlan_alt_mode == BITBANG) {
    if ((TIM12->SR & TIM_SR_UIF) && (gmlan_sendmax != -1)) {
      int read = get_gpio_input(GPIOB, 12);
      if (gmlan_silent_count < REQUIRED_SILENT_TIME) {
        if (read == 0) {
          gmlan_silent_count = 0;
        } else {
          gmlan_silent_count++;
        }
      } else {
        bool retry = 0;
        // in send loop
        if ((gmlan_sending > 0) &&  // not first bit
           ((read == 0) && (pkt_stuffed[gmlan_sending-1] == 1)) &&  // bus wrongly dominant
           (gmlan_sending != (gmlan_sendmax - 11))) {    //not ack bit
          print("GMLAN ERR: bus driven at ");
          puth(gmlan_sending);
          print("\n");
          retry = 1;
        } else if ((read == 1) && (gmlan_sending == (gmlan_sendmax - 11))) {    // recessive during ACK
          print("GMLAN ERR: didn't recv ACK\n");
          retry = 1;
        } else {
          // do not retry
        }
        if (retry) {
          // reset sender (retry after 7 silent)
          set_bitbanged_gmlan(1); // recessive
          gmlan_silent_count = 0;
          gmlan_sending = 0;
          gmlan_fail_count++;
          if (gmlan_fail_count == MAX_FAIL_COUNT) {
            print("GMLAN ERR: giving up send\n");
            gmlan_send_ok = false;
          }
        } else {
          set_bitbanged_gmlan(pkt_stuffed[gmlan_sending]);
          gmlan_sending++;
        }
      }
      if ((gmlan_sending == gmlan_sendmax) || (gmlan_fail_count == MAX_FAIL_COUNT)) {
        set_bitbanged_gmlan(1); // recessive
        set_gpio_mode(GPIOB, 13, MODE_INPUT);
        register_clear_bits(&(TIM12->DIER), TIM_DIER_UIE); // No update interrupt
        register_set(&(TIM12->CR1), 0U, 0x3FU); // Disable timer
        gmlan_sendmax = -1;   // exit
      }
    }
  } else if (gmlan_alt_mode == GPIO_SWITCH) {
    if ((TIM12->SR & TIM_SR_UIF) && (gmlan_switch_below_timeout != -1)) {
      if ((can_timeout_counter == 0) && gmlan_switch_timeout_enable) {
        //it has been more than 1 second since timeout was reset; disable timer and restore the GMLAN output
        set_gpio_output(GPIOB, 13, GMLAN_LOW);
        gmlan_switch_below_timeout = -1;
        gmlan_timeout_counter = GMLAN_TICKS_PER_TIMEOUT_TICKLE;
        gmlan_alt_mode = DISABLED;
      }
      else {
        can_timeout_counter--;
        if (gmlan_timeout_counter == 0) {
          //Send a 1 (bus low) every 15ms to reset the GMLAN transceivers timeout
          gmlan_timeout_counter = GMLAN_TICKS_PER_TIMEOUT_TICKLE;
          set_gpio_output(GPIOB, 13, GMLAN_LOW);
        }
        else {
          set_gpio_output(GPIOB, 13, inverted_bit_to_send);
          gmlan_timeout_counter--;
        }
      }
    }
  } else {
    // Invalid GMLAN mode. Do not put a print statement here, way too fast to keep up with
  }
  TIM12->SR = 0;
}

bool bitbang_gmlan(CANPacket_t *to_bang) {
  gmlan_send_ok = true;
  gmlan_alt_mode = BITBANG;

#ifndef STM32H7
  if (gmlan_sendmax == -1) {
    int len = get_bit_message(pkt_stuffed, to_bang);
    gmlan_fail_count = 0;
    gmlan_silent_count = 0;
    gmlan_sending = 0;
    gmlan_sendmax = len;
    // setup for bitbang loop
    set_bitbanged_gmlan(1); // recessive
    set_gpio_mode(GPIOB, 13, MODE_OUTPUT);

    // 33kbps
    setup_timer();
  }
#else
  UNUSED(to_bang);
#endif
  return gmlan_send_ok;
}
