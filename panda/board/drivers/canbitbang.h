#define MAX_BITS_CAN_PACKET (200)

// returns out_len
int do_bitstuff(char *out, char *in, int in_len) {
  int last_bit = -1;
  int bit_cnt = 0;
  int j = 0;
  for (int i = 0; i < in_len; i++) {
    char bit = in[i];
    out[j++] = bit;

    // do the stuffing
    if (bit == last_bit) {
      bit_cnt++;
      if (bit_cnt == 5) {
        // 5 in a row the same, do stuff
        last_bit = !bit;
        out[j++] = last_bit;
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
    if (in[i] ^ ((crc>>15)&1)) {
      crc = crc ^ 0x4599;
    }
    crc &= 0x7fff;
  }
  for (int i = 14; i >= 0; i--) {
    in[in_len++] = (crc>>i)&1;
  }
  return in_len;
}

int append_bits(char *in, int in_len, char *app, int app_len) {
  for (int i = 0; i < app_len; i++) {
    in[in_len++] = app[i];
  }
  return in_len;
}

int append_int(char *in, int in_len, int val, int val_len) {
  for (int i = val_len-1; i >= 0; i--) {
    in[in_len++] = (val&(1<<i)) != 0;
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
  
  if (to_bang->RIR & 4) {
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

// hardware stuff below this line

#ifdef PANDA

void set_bitbanged_gmlan(int val) {
  if (val) {
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
  if (TIM4->SR & TIM_SR_UIF && gmlan_sendmax != -1) {
    int read = get_gpio_input(GPIOB, 12);
    if (gmlan_silent_count < REQUIRED_SILENT_TIME) {
      if (read == 0) {
        gmlan_silent_count = 0;
      } else {
        gmlan_silent_count++;
      }
    } else if (gmlan_silent_count == REQUIRED_SILENT_TIME) {
      int retry = 0;
      // in send loop
      if (gmlan_sending > 0 &&  // not first bit
         (read == 0 && pkt_stuffed[gmlan_sending-1] == 1) &&  // bus wrongly dominant
         gmlan_sending != (gmlan_sendmax-11)) {    //not ack bit
        puts("GMLAN ERR: bus driven at ");
        puth(gmlan_sending);
        puts("\n");
        retry = 1;
      } else if (read == 1 && gmlan_sending == (gmlan_sendmax-11)) {    // recessive during ACK
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
    if (gmlan_sending == gmlan_sendmax || gmlan_fail_count == MAX_FAIL_COUNT) {
      set_bitbanged_gmlan(1); // recessive
      set_gpio_mode(GPIOB, 13, MODE_INPUT);
      TIM4->DIER = 0;  // no update interrupt
      TIM4->CR1 = 0;   // disable timer
      gmlan_sendmax = -1;   // exit
    }
  }
  TIM4->SR = 0;
}

void bitbang_gmlan(CAN_FIFOMailBox_TypeDef *to_bang) {
  // TODO: make failure less silent
  if (gmlan_sendmax != -1) return;

  int len = get_bit_message(pkt_stuffed, to_bang);
  gmlan_fail_count = 0;
  gmlan_silent_count = 0;
  gmlan_sending = 0;
  gmlan_sendmax = len;

  // setup for bitbang loop
  set_bitbanged_gmlan(1); // recessive
  set_gpio_mode(GPIOB, 13, MODE_OUTPUT);

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

#endif

