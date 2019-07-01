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
  int crc = 0;
  for (int i = 0; i < in_len; i++) {
    crc <<= 1;
    if ((in[i] ^ ((crc >> 15) & 1)) != 0) {
      crc = crc ^ 0x4599;
    }
    crc &= 0x7fff;
  }
  for (int i = 14; i >= 0; i--) {
    in[in_len] = (crc>>i)&1;
    in_len++;
  }
  return in_len;
}

int append_bits(char *in, int in_len, char *app, int app_len) {
  for (int i = 0; i < app_len; i++) {
    in[in_len] = app[i];
    in_len++;
  }
  return in_len;
}

int append_int(char *in, int in_len, int val, int val_len) {
  for (int i = val_len-1; i >= 0; i--) {
    in[in_len] = (val&(1<<i)) != 0;
    in_len++;
  }
  return in_len;
}

int get_bit_message(char *out, CAN_FIFOMailBox_TypeDef *to_bang) {
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
  int dlc_len = to_bang->RDTR & 0xF;
  len = append_int(pkt, len, 0, 1);    // Start-of-frame

  if ((to_bang->RIR & 4) != 0) {
    // extended identifier
    len = append_int(pkt, len, to_bang->RIR >> 21, 11);  // Identifier
    len = append_int(pkt, len, 3, 2);    // SRR+IDE
    len = append_int(pkt, len, (to_bang->RIR >> 3) & ((1<<18)-1), 18);  // Identifier
    len = append_int(pkt, len, 0, 3);    // RTR+r1+r0
  } else {
    // standard identifier
    len = append_int(pkt, len, to_bang->RIR >> 21, 11);  // Identifier
    len = append_int(pkt, len, 0, 3);    // RTR+IDE+reserved
  }

  len = append_int(pkt, len, dlc_len, 4);    // Data length code

  // append data
  for (int i = 0; i < dlc_len; i++) {
    unsigned char dat = ((unsigned char *)(&(to_bang->RDLR)))[i];
    len = append_int(pkt, len, dat, 8);
  }

  // append crc
  len = append_crc(pkt, len);

  // do bitstuffing
  len = do_bitstuff(out, pkt, len);

  // append footer
  len = append_bits(out, len, footer, sizeof(footer));
  return len;
}

void setup_timer4(void) {
  // setup
  TIM4->PSC = 48-1;          // tick on 1 us
  TIM4->CR1 = TIM_CR1_CEN;   // enable
  TIM4->ARR = 30-1;          // 33.3 kbps

  // in case it's disabled
  NVIC_EnableIRQ(TIM4_IRQn);

  // run the interrupt
  TIM4->DIER = TIM_DIER_UIE; // update interrupt
  TIM4->SR = 0;
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

  setup_timer4();

  inverted_bit_to_send = GMLAN_LOW; //We got initialized, set the output low
}

void set_gmlan_digital_output(int to_set) {
  inverted_bit_to_send = to_set;
  /*
  puts("Writing ");
  puth(inverted_bit_to_send);
  puts("\n");
  */
}

void reset_gmlan_switch_timeout(void) {
  can_timeout_counter = GMLAN_TICKS_PER_SECOND;
  gmlan_switch_below_timeout = 1;
  gmlan_alt_mode = GPIO_SWITCH;
}

void set_bitbanged_gmlan(int val) {
  if (val != 0) {
    GPIOB->ODR |= (1 << 13);
  } else {
    GPIOB->ODR &= ~(1 << 13);
  }
}

char pkt_stuffed[MAX_BITS_CAN_PACKET];
int gmlan_sending = -1;
int gmlan_sendmax = -1;

int gmlan_silent_count = 0;
int gmlan_fail_count = 0;
#define REQUIRED_SILENT_TIME 10
#define MAX_FAIL_COUNT 10

void TIM4_IRQHandler(void) {
  if (gmlan_alt_mode == BITBANG) {
    if ((TIM4->SR & TIM_SR_UIF) && (gmlan_sendmax != -1)) {
      int read = get_gpio_input(GPIOB, 12);
      if (gmlan_silent_count < REQUIRED_SILENT_TIME) {
        if (read == 0) {
          gmlan_silent_count = 0;
        } else {
          gmlan_silent_count++;
        }
      } else if (gmlan_silent_count == REQUIRED_SILENT_TIME) {
        bool retry = 0;
        // in send loop
        if ((gmlan_sending > 0) &&  // not first bit
           ((read == 0) && (pkt_stuffed[gmlan_sending-1] == 1)) &&  // bus wrongly dominant
           (gmlan_sending != (gmlan_sendmax - 11))) {    //not ack bit
          puts("GMLAN ERR: bus driven at ");
          puth(gmlan_sending);
          puts("\n");
          retry = 1;
        } else if ((read == 1) && (gmlan_sending == (gmlan_sendmax - 11))) {    // recessive during ACK
          puts("GMLAN ERR: didn't recv ACK\n");
          retry = 1;
        }
        if (retry) {
          // reset sender (retry after 7 silent)
          set_bitbanged_gmlan(1); // recessive
          gmlan_silent_count = 0;
          gmlan_sending = 0;
          gmlan_fail_count++;
          if (gmlan_fail_count == MAX_FAIL_COUNT) {
            puts("GMLAN ERR: giving up send\n");
          }
        } else {
          set_bitbanged_gmlan(pkt_stuffed[gmlan_sending]);
          gmlan_sending++;
        }
      }
      if ((gmlan_sending == gmlan_sendmax) || (gmlan_fail_count == MAX_FAIL_COUNT)) {
        set_bitbanged_gmlan(1); // recessive
        set_gpio_mode(GPIOB, 13, MODE_INPUT);
        TIM4->DIER = 0;  // no update interrupt
        TIM4->CR1 = 0;   // disable timer
        gmlan_sendmax = -1;   // exit
      }
    }
    TIM4->SR = 0;
  } //bit bang mode

  else if (gmlan_alt_mode == GPIO_SWITCH) {
    if ((TIM4->SR & TIM_SR_UIF) && (gmlan_switch_below_timeout != -1)) {
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
    TIM4->SR = 0;
  } //gmlan switch mode
}

void bitbang_gmlan(CAN_FIFOMailBox_TypeDef *to_bang) {
  gmlan_alt_mode = BITBANG;
  // TODO: make failure less silent
  if (gmlan_sendmax == -1) {

    int len = get_bit_message(pkt_stuffed, to_bang);
    gmlan_fail_count = 0;
    gmlan_silent_count = 0;
    gmlan_sending = 0;
    gmlan_sendmax = len;

    // setup for bitbang loop
    set_bitbanged_gmlan(1); // recessive
    set_gpio_mode(GPIOB, 13, MODE_OUTPUT);

    setup_timer4();
  }
}

